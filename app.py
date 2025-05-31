from flask import Flask, render_template, request, jsonify, send_from_directory, session
import chess
import random
import requests
import os
import logging
import uuid
from datetime import timedelta

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key_for_sessions')  # For session security
app.permanent_session_lifetime = timedelta(days=7)  # Session lasts for 7 days

# Llama 3.3 70B API settings
LLAMA_API_KEY = os.environ.get('LLAMA_API_KEY', '')
LLAMA_API_URL = "https://api.venice.ai/api/v1/chat/completions"  # Corrected API endpoint

if not LLAMA_API_KEY:
    raise RuntimeError("LLAMA_API_KEY is not set in environment variables or .env file.")

# Store games in memory (in a production app, you'd use a database)
games = {}

# Get or create a session ID for the user
def get_session_id():
    if 'user_id' not in session:
        session.permanent = True
        session['user_id'] = str(uuid.uuid4())
        session['active_game_id'] = None
    return session['user_id']

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    audio_dir = os.path.join(app.root_path, 'audio')
    return send_from_directory(audio_dir, filename)

@app.route('/')
def index():
    # Ensure user has a session
    user_id = get_session_id()
    return render_template('index.html')

@app.route('/session_status', methods=['GET'])
def session_status():
    """Return the user's session information including active game."""
    user_id = get_session_id()
    active_game_id = session.get('active_game_id')
    
    # Get the actual game data if it exists
    game_data = None
    if active_game_id and active_game_id in games:
        board = games[active_game_id]['board']
        game_data = {
            'game_id': active_game_id,
            'fen': board.fen(),
            'mode': games[active_game_id]['mode'],
            'moves': [move.uci() for move in board.move_stack]
        }
    
    return jsonify({
        'user_id': user_id,
        'active_game_id': active_game_id,
        'game_data': game_data
    })

@app.route('/new_game', methods=['POST'])
def new_game():
    user_id = get_session_id()
    data = request.get_json() or {}
    mode = data.get('mode', 'regular')  # 'regular' or 'ai'
    ai_side = data.get('ai_side', 'black')  # Optional: 'white' or 'black', default is 'black'

    # Generate a new game ID
    game_id = str(random.randint(1000, 9999))

    logger.info(f"Creating new game with ID {game_id}, mode: {mode} for user {user_id}")

    from datetime import datetime
    board = chess.Board()
    games[game_id] = {
        'board': board,
        'mode': mode,
        'user_id': user_id,
        'created_at': datetime.now().isoformat()
    }

    # If AI is to play as White, make the first move
    ai_move = None
    if mode == 'ai' and ai_side == 'white':
        try:
            ai_move = get_ai_move(board)
            if ai_move:
                board.push(ai_move)
                logger.info(f"AI (White) moved: {ai_move.uci()}")
            else:
                logger.error("AI did not return a valid move as White")
        except Exception as e:
            logger.error(f"Error making AI move as White: {str(e)}")

    # Update the user's session with the active game
    session['active_game_id'] = game_id

    return jsonify({
        'game_id': game_id,
        'fen': games[game_id]['board'].fen(),
        'mode': mode,
        'ai_move': ai_move.uci() if ai_move else None
    })

@app.route('/make_move', methods=['POST'])
def make_move():
    user_id = get_session_id()
    data = request.get_json() or {}
    game_id = data.get('game_id')
    move_uci = data.get('move')
    promotion = data.get('promotion')
    
    logger.info(f"Move request for game {game_id}: {move_uci}, promotion: {promotion} from user {user_id}")
    
    if game_id not in games:
        logger.error(f"Game not found: {game_id}")
        return jsonify({'error': f'Game not found with ID: {game_id}'}), 404
    
    # Verify the user owns this game or it's a shared game
    if games[game_id].get('user_id') != user_id and not games[game_id].get('shared', False):
        logger.warning(f"User {user_id} attempted to access game {game_id} owned by {games[game_id].get('user_id')}")
        return jsonify({'error': 'You do not have permission to access this game'}), 403
    
    board = games[game_id]['board']
    mode = games[game_id]['mode']
    
    # Process player's move
    try:
        move = chess.Move.from_uci((move_uci or '') + (promotion or ''))
        if move not in board.legal_moves:
            logger.warning(f"Illegal move attempted: {move_uci}, promotion: {promotion}")
            return jsonify({'error': 'Illegal move'}), 400
        board.push(move)
        logger.info(f"Player moved: {move_uci}, promotion: {promotion}")
    except Exception as e:
        logger.error(f"Error processing move: {str(e)}")
        return jsonify({'error': str(e)}), 400
    
    # Check game state after player's move
    game_status = get_game_status(board)
    if game_status != 'ongoing':
        logger.info(f"Game {game_id} ended with status: {game_status}")
        return jsonify({
            'fen': board.fen(),
            'status': game_status,
            'last_move': move_uci,
            'moves': [move.uci() for move in board.move_stack]
        })
    
    # If AI mode and game is still ongoing, make AI move
    ai_move = None
    if mode == 'ai' and not board.is_game_over():
        try:
            # Get AI move and make it on the board
            ai_move = get_ai_move(board)
            if ai_move:
                board.push(ai_move)
                logger.info(f"AI moved: {ai_move.uci()}")
            else:
                logger.error("AI did not return a valid move")
                return jsonify({'error': 'AI did not return a valid move'}), 500
        except Exception as e:
            logger.error(f"Error making AI move: {str(e)}")
            return jsonify({'error': f'AI move error: {str(e)}'}), 500
    
    game_status = get_game_status(board)
    
    return jsonify({
        'fen': board.fen(),
        'status': game_status,
        'last_move': move_uci,
        'ai_move': ai_move.uci() if ai_move else None,
        'moves': [move.uci() for move in board.move_stack]
    })

@app.route('/restart_game', methods=['POST'])
def restart_game():
    user_id = get_session_id()
    data = request.json or {}
    game_id = data.get('game_id')
    mode = data.get('mode', 'regular')
    
    logger.info(f"Restarting game {game_id} in {mode} mode for user {user_id}")
    
    # Handle the case where we're restarting a non-existent game
    if game_id == 'new' or game_id not in games:
        game_id = str(random.randint(1000, 9999))
        logger.info(f"Creating new game with ID {game_id} instead of restarting")
    elif games[game_id].get('user_id') != user_id and not games[game_id].get('shared', False):
        # User doesn't own this game, create a new one instead
        game_id = str(random.randint(1000, 9999))
        logger.info(f"User {user_id} attempted to restart game {game_id} they don't own, creating new game instead")
    
    from datetime import datetime
    games[game_id] = {
        'board': chess.Board(),
        'mode': mode,
        'user_id': user_id,
        'created_at': datetime.now().isoformat()
    }
    
    # Update the user's session with the active game
    session['active_game_id'] = game_id
    
    return jsonify({
        'game_id': game_id,
        'fen': games[game_id]['board'].fen(),
        'mode': mode
    })

def get_game_status(board):
    if board.is_checkmate():
        return 'checkmate'
    elif board.is_stalemate():
        return 'stalemate'
    elif board.is_insufficient_material():
        return 'draw_insufficient_material'
    elif board.is_fifty_moves():
        return 'draw_fifty_moves'
    elif board.is_repetition():
        return 'draw_repetition'
    else:
        return 'ongoing'

def get_ai_move(board):
    """Get a move exclusively from Llama."""
    return get_llama_move(board)

def get_llama_move(board):
    """Get a move from the Llama 3.3 70B AI model with human-like grandmaster play."""
    try:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Create board representation for better context
        fen = board.fen()
        logger.info(f"Using board FEN for Llama: {fen}")
        board_ascii = str(board)
        
        # Convert legal moves to a readable format for the model with enhanced descriptions
        legal_moves_descriptions = []
        for move in legal_moves:
            piece = board.piece_at(move.from_square)
            piece_type = piece.symbol().upper() if piece.color == chess.WHITE else piece.symbol().lower()
            
            # Enhanced description for captures
            is_capture = board.is_capture(move)
            capture = "x" if is_capture else "-"
            target_piece = board.piece_at(move.to_square)
            
            # Make captures more explicit
            if is_capture and target_piece:
                captured_piece = target_piece.symbol().upper() if target_piece.color == chess.WHITE else target_piece.symbol().lower()
                target_desc = f"CAPTURES {captured_piece}"
            else:
                target_desc = "empty"
                
            # Add material value indicators for captures to encourage material gain
            if is_capture and target_piece:
                piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
                piece_symbol = target_piece.symbol().lower()
                value = piece_values.get(piece_symbol, 0)
                move_text = f"{move.uci()} ({piece_type} {capture} {target_desc}, value: +{value})"
            else:
                # Add positional context for non-captures
                from_file, from_rank = move.from_square % 8, move.from_square // 8
                to_file, to_rank = move.to_square % 8, move.to_square // 8
                
                # Check if move is toward center (e4, d4, e5, d5)
                center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
                center_proximity = "→center" if move.to_square in center_squares else ""
                
                # Check if the move is a development move for minor pieces
                development = ""
                if piece_type.lower() in ['n', 'b'] and from_rank in [0, 7]:
                    development = "→development"
                
                # Check if it's a castling move
                castling = ""
                if piece_type.lower() == 'k' and abs(from_file - to_file) > 1:
                    castling = "→castling"
                
                # Add this context to non-capture moves
                positional_info = " ".join(filter(None, [center_proximity, development, castling]))
                if positional_info:
                    move_text = f"{move.uci()} ({piece_type} {capture} {target_desc}, {positional_info})"
                else:
                    move_text = f"{move.uci()} ({piece_type} {capture} {target_desc})"
                
            legal_moves_descriptions.append(move_text)
        
        # Generate phase-appropriate advice
        phase_advice = ""
        piece_count = len(board.piece_map())
        if piece_count >= 28:  # Opening
            phase_advice = "OPENING PHASE: Focus on piece development, center control, and king safety (castling)."
        elif piece_count >= 15:  # Middlegame
            phase_advice = "MIDDLEGAME PHASE: Create attacking plans, coordinate pieces, and find tactical opportunities."
        else:  # Endgame
            phase_advice = "ENDGAME PHASE: Centralize your king, promote pawns, and simplify when ahead in material."
        
        # Track move history to help avoid repetitions
        move_history = [move.uci() for move in board.move_stack]
        repeated_patterns = find_repeated_patterns(move_history)
        
        # Enhanced prompt with clear instructions for human-like grandmaster play
        prompt = f"""As a human chess grandmaster, analyze this position deeply and choose your next move:

Board:
{board_ascii}

FEN: {fen}
Turn: {'White' if board.turn == chess.WHITE else 'Black'}

{phase_advice}

Legal Moves (you must pick only from these):
{', '.join(legal_moves_descriptions)}

Previous moves: {' '.join(move_history) if move_history else 'None'}
Note: Avoid repeating moves or positions. Patterns to avoid: {repeated_patterns}

Think like a human grandmaster:
1. Consider multiple candidate moves, not just the first one you see
2. Balance material gains with positional understanding
3. Think about your opponent's threats and plans
4. Use different pieces, not just focusing on one
5. Consider long-term pawn structure
6. In equal positions, create imbalances to outplay your opponent

Reply ONLY with your chosen move in UCI format (e.g. "e2e4").
"""
        
        # Call the Llama API with chess-focused parameters
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLAMA_API_KEY}"
        }
        
        data = {
            "model": "llama-3.3-70b",
            "messages": [
                {
                    "role": "system", 
                    "content": (
                        "You are a human chess grandmaster with a creative, positional style. "
                        "You always consider at least 3 different candidate moves before choosing. "
                        "You avoid repeating the same moves or patterns, and you do not always play the same opening or reply. "
                        "You enjoy variety and sometimes play less common but still strong moves. "
                        "Think about the whole position, using multiple pieces in coordination. "
                        "Avoid fixating on just one piece. Respond with only your chosen move in UCI notation."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 20,
            "temperature": 0.5,  # Slightly higher for more diverse play
            "top_p": 0.95
        }
        
        # Make the API call
        logger.info(f"Calling Llama API for chess analysis")
        response = requests.post(LLAMA_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code != 200:
            raise ValueError(f"Lamma API error: {response.status_code}")
        
        # Parse the response
        response_data = response.json()
        move_text = response_data["choices"][0]["message"]["content"].strip().lower()
        logger.info(f"Lamma suggested: {move_text}")
        
        # Extract the UCI move from the text
        import re
        move_pattern = re.compile(r'([a-h][1-8][a-h][1-8][qrbnk]?)')
        match = move_pattern.search(move_text)
        
        if match:
            suggested_move = match.group(1)
            for move in legal_moves:
                if move.uci() == suggested_move:
                    logger.info(f"Using Lamma's move: {suggested_move}")
                    return move
            logger.warning(f"Lamma suggested illegal move: {suggested_move}, using simple fallback.")
            return get_smart_fallback_move(board, legal_moves)
        else:
            logger.warning(f"Could not extract a valid move from: {move_text}")
            return get_smart_fallback_move(board, legal_moves)
    except Exception as e:
        logger.error(f"Error in get_llama_move: {str(e)}")
        # Ensure legal_moves is defined for fallback
        try:
            legal_moves = list(board.legal_moves)
        except Exception:
            legal_moves = []
        return get_smart_fallback_move(board, legal_moves)

def get_smart_fallback_move(board, legal_moves):
    """Strategic fallback when Llama fails to provide a valid move."""
    logger.warning("Using strategic fallback move selection")
    
    if not legal_moves:
        return None
    
    # Convert to list if it's not already
    legal_moves = list(legal_moves)
    
    # Prioritize moves by type
    for move in legal_moves:
        # Prioritize checkmate
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()
    
    # Avoid moving queen to squares attacked by opponent
    safe_moves = []
    unsafe_moves = []
    queen_moves = []
    attack_moves = []
    develop_moves = []
    
    # Identify the player's color
    current_color = board.turn
    
    # First, categorize moves for better decision making
    for move in legal_moves:
        piece = board.piece_at(move.from_square)
        
        # Skip if piece is None (shouldn't happen, but just in case)
        if piece is None:
            continue
            
        # Check if this is a queen move
        is_queen_move = piece.piece_type == chess.QUEEN
        
        # Track queen moves separately
        if is_queen_move:
            queen_moves.append(move)
            
        # Make the move to analyze resulting position
        board.push(move)
        
        # Check if the piece would be captured after this move
        is_safe = True
        target_square = move.to_square
        
        # Create a set of squares attacked by opponent
        opponent_attacks = set()
        for square in chess.SQUARES:
            if board.is_attacked_by(not current_color, square):
                opponent_attacks.add(square)
        
        # If we just moved a piece to a square that's under attack
        if target_square in opponent_attacks:
            # It's unsafe if:
            # 1. It's a queen move to an attacked square
            # 2. The attacker is of lower value than our piece
            if is_queen_move or piece.piece_type in [chess.WHITE, chess.ROOK]:
                # Check what's attacking this square (to avoid trading queen for pawn, etc.)
                attackers = []
                for attack_sq in chess.SQUARES:
                    attack_piece = board.piece_at(attack_sq)
                    if attack_piece and attack_piece.color != current_color:
                        if board.is_legal(chess.Move(attack_sq, target_square)):
                            attackers.append(attack_piece.piece_type)
                
                # Determine if trade is unfavorable
                piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
                moved_value = piece_values.get(piece.piece_type, 0)
                
                # If any attacker is of lower value than our piece, it's unsafe
                for attacker in attackers:
                    if piece_values.get(attacker, 0) < moved_value:
                        is_safe = False
                        break
            
        # Undo the move
        board.pop()
        
        # Add to appropriate list
        if is_safe:
            safe_moves.append(move)
            
            # Check if it's a capture move
            if board.is_capture(move):
                attack_moves.append(move)
                
            # Check if it's a development move (knights or bishops from starting position)
            if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                from_rank = move.from_square // 8
                if (current_color == chess.WHITE and from_rank == 0) or \
                   (current_color == chess.BLACK and from_rank == 7):
                    develop_moves.append(move)
        else:
            unsafe_moves.append(move)
    
    # Prioritize high-value captures among safe moves
    for move in safe_moves:
        if board.is_capture(move):
            target_square = move.to_square
            captured_piece = board.piece_at(target_square)
            if captured_piece and captured_piece.piece_type in [chess.QUEEN, chess.ROOK]:
                return move
    
    # Prioritize any safe capture
    if attack_moves:
        return random.choice(attack_moves)
    
    # In opening, prioritize development
    piece_count = len(board.piece_map())
    if piece_count >= 28 and develop_moves:  # Opening phase
        return random.choice(develop_moves)
    
    # Prioritize check among safe moves
    for move in safe_moves:
        board.push(move)
        if board.is_check():
            board.pop()
            return move
        board.pop()
    
    # If we have safe moves, prefer those
    if safe_moves:
        # Final fallback: pick a random safe move with slight bias toward center
        center_moves = []
        other_moves = []
        
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        for move in safe_moves:
            if move.to_square in center_squares:
                center_moves.append(move)
            else:
                other_moves.append(move)
        
        if center_moves and random.random() < 0.7:  # 70% chance to prefer center moves
            return random.choice(center_moves)
        else:
            return random.choice(safe_moves)
    
    # If no safe moves, pick the least bad move
    # Avoid queen captures if possible
    non_queen_moves = [m for m in legal_moves if board.piece_at(m.from_square).piece_type != chess.QUEEN]
    if non_queen_moves:
        return random.choice(non_queen_moves)
    
    # Absolutely last resort - any legal move
    return random.choice(legal_moves)

def find_repeated_patterns(move_history):
    """Identify repetitive patterns in the move history."""
    if len(move_history) < 4:
        return "None detected yet"
    
    # Look for piece moving back and forth
    repeated_moves = []
    for i in range(len(move_history) - 2):
        # Check if a piece moved from A to B, then back to A
        move1 = move_history[i]
        move2 = move_history[i+1]
        if move1[:2] == move2[2:4] and move1[2:4] == move2[:2]:
            repeated_moves.append(f"{move1}-{move2}")
    
    # Look for longer patterns (3-4 moves repeating)
    for pattern_length in range(2, min(6, len(move_history) // 2)):
        for i in range(len(move_history) - pattern_length * 2):
            pattern1 = move_history[i:i+pattern_length]
            pattern2 = move_history[i+pattern_length:i+pattern_length*2]
            if pattern1 == pattern2:
                repeated_moves.append("Pattern: " + "-".join(pattern1))
    
    # Track positions that recur
    position_history = {}
    test_board = chess.Board()
    for i, move in enumerate(move_history):
        try:
            test_board.push_uci(move)
            fen_key = test_board.fen().split(' ')[0]  # Just material position without move counters
            if fen_key in position_history:
                position_history[fen_key].append(i)
            else:
                position_history[fen_key] = [i]
        except:
            pass
    
    # Find positions that occurred multiple times
    repeated_positions = [key for key, indices in position_history.items() if len(indices) >= 2]
    if repeated_positions:
        repeated_moves.append(f"{len(repeated_positions)} repeated positions")
    
    return "; ".join(repeated_moves) if repeated_moves else "None detected"

if __name__ == '__main__':
    app.run(debug=True)
