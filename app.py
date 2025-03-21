from flask import Flask, render_template, request, jsonify, send_from_directory
import chess
import random
import requests
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Llama 3.3 70B API settings
LLAMA_API_KEY = "-Y3up9vlEXoVFf1ZsrXhB4rbPXd8V6ywgiSZziI3bR"
LLAMA_API_URL = "https://api.venice.ai/api/v1/chat/completions"  # Corrected API endpoint

# Store games in memory (in a production app, you'd use a database)
games = {}

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    audio_dir = os.path.join(app.root_path, 'audio')
    return send_from_directory(audio_dir, filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new_game', methods=['POST'])
def new_game():
    game_id = str(random.randint(1000, 9999))
    mode = request.json.get('mode', 'regular')  # 'regular' or 'ai'
    
    logger.info(f"Creating new game with ID {game_id}, mode: {mode}")
    
    games[game_id] = {
        'board': chess.Board(),
        'mode': mode
    }
    
    return jsonify({
        'game_id': game_id,
        'fen': games[game_id]['board'].fen(),
        'mode': mode
    })

@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.json
    game_id = data.get('game_id')
    move_uci = data.get('move')
    promotion = data.get('promotion')
    
    logger.info(f"Move request for game {game_id}: {move_uci}, promotion: {promotion}")
    
    if game_id not in games:
        logger.error(f"Game not found: {game_id}")
        return jsonify({'error': f'Game not found with ID: {game_id}'}), 404
    
    board = games[game_id]['board']
    mode = games[game_id]['mode']
    
    # Process player's move
    try:
        move = chess.Move.from_uci(move_uci + (promotion if promotion else ''))
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
            'last_move': move_uci
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
        'ai_move': ai_move.uci() if ai_move else None
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
    return get_llama_move(board)

def get_llama_move(board):
    try:
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Let's use a deterministic evaluation approach
        if random.random() < 0.7:  # Use engine-based evaluation 70% of the time for consistent quality
            return get_engine_evaluated_move(board, legal_moves)
        
        # Only use Llama API 30% of the time
        # Define piece name map
        piece_name_map = {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
            chess.KING: "king"
        }
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Just return first move if only one is available
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Create board representation for better context with coordinates
        fen = board.fen()
        logger.info(f"Using board FEN for Llama: {fen}")
        board_ascii = str(board)
        # Add column coordinates at the bottom
        # Form a list of legal moves with descriptions to help the model understand
        legal_moves_descriptions = []
        for move in legal_moves:
            piece = board.piece_at(move.from_square)
            piece_type = piece.symbol().upper() if piece.color == chess.WHITE else piece.symbol().lower()
            capture = "x" if board.is_capture(move) else "-"
            target_piece = board.piece_at(move.to_square)
            target_desc = target_piece.symbol() if target_piece else "empty"
            legal_moves_descriptions.append(f"{move.uci()} ({piece_type} {capture} {target_desc})")
        
        # Add move history to context
        move_history = []
        for move in board.move_stack:
            move_history.append(move.uci())
        
        # Identify repeated moves to avoid them
        repeated_patterns = find_repeated_patterns(move_history)
        
        # Enhanced prompt with move history and instruction to avoid repetition
        legal_moves_uci = [m.uci() for m in legal_moves]
        prompt = f"""As a grandmaster chess engine, you must pick the best move ONLY from the list below:
Legal Moves: {', '.join(legal_moves_uci)}

Position:
FEN: {fen}
Board: 
{board_ascii}

Move History: {' '.join(move_history)}

IMPORTANT: Avoid repeating moves or positions. Do not fall into repetitive patterns.
Previous repetitive patterns detected: {repeated_patterns}

Consider:
1. Developing new pieces rather than moving the same ones repeatedly
2. Controlling the center
3. Protecting your king
4. Creating threats and attacking opportunities
5. Breaking repetitive patterns

Respond ONLY with the single best move in UCI format (e.g., "e2e4")."""

        # Call the Llama API with more specific chess parameters
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLAMA_API_KEY}"
        }
        data = {
            "model": "llama-3.3-70b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a high-level chess engine. Always calculate at least 12 moves ahead. Avoid quick or dubious sacrifices. Provide your single best legal move."
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,  # Increased for deeper analysis
            "temperature": 0.0,
            "top_p": 1.0
        }
        logger.info("Calling Llama API for chess analysis")
        response = requests.post(LLAMA_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code != 200:
            raise ValueError(f"Llama API error: {response.status_code}")
        
        # Parse the response
        response_data = response.json()
        move_text = response_data["choices"][0]["message"]["content"].strip().lower()
        logger.info(f"Llama suggested: {move_text}")
        
        # Try to find any UCI pattern in the response
        import re
        move_pattern = re.compile(r'([a-h][1-8][a-h][1-8][qrbnk]?)')
        match = move_pattern.search(move_text)
        
        if match:
            suggested_move = match.group(1)
            # Check legality
            for m in legal_moves:
                if m.uci() == suggested_move:
                    return m
            logger.warning(f"Llama suggested illegal move: {suggested_move}, retrying.")
            return retry_llama_move(board, legal_moves, previous_suggestion=suggested_move)
        else:
            logger.warning("No valid UCI pattern found in Llama response, retrying.")
            return retry_llama_move(board, legal_moves)
    except Exception as e:
        logger.error(f"Error in get_llama_move: {str(e)}")
        return get_engine_evaluated_move(board, legal_moves)

def get_engine_evaluated_move(board, legal_moves):
    """Use a deterministic engine evaluation approach instead of relying on Llama."""
    logger.info("Using engine evaluation for move selection")
    
    # Copy board for evaluation
    test_board = board.copy()
    
    # Dictionary to store evaluated moves
    move_scores = {}
    
    # Material values (standard chess piece values in centipawns)
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320, 
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    # Piece positioning values - encourage piece development and center control
    piece_position_values = {
        chess.PAWN: [  # Pawns are valued higher in the center and promoted ranks
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ],
        chess.KNIGHT: [  # Knights are best in the center, bad on the rim
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ],
        chess.BISHOP: [  # Bishops want open diagonals
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5,  5,  5,  5,  5,-10,
            -10,  0,  5,  0,  0,  5,  0,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ],
        chess.ROOK: [  # Rooks want open files
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ],
        chess.QUEEN: [  # Queen combines bishop and rook mobility
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ],
        chess.KING: [  # King wants safety in the opening/middlegame
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]
    }
    
    # Flip tables for black pieces
    flipped_piece_position_values = {}
    for piece_type, table in piece_position_values.items():
        flipped_table = list(reversed(table))
        flipped_piece_position_values[piece_type] = flipped_table
    
    # Calculate opening/middlegame/endgame phase
    def get_game_phase(board):
        piece_count = len(board.piece_map())
        if piece_count >= 26:  # Most/all pieces still on board
            return "opening"
        elif piece_count >= 12:  # Some pieces traded
            return "middlegame"
        else:  # Few pieces left
            return "endgame"
    
    game_phase = get_game_phase(board)
    
    # Adjust king positioning for endgame
    if game_phase == "endgame":
        # In endgame, king should move to center
        piece_position_values[chess.KING] = [
            -50,-40,-30,-20,-20,-30,-40,-50,
            -30,-20,-10,  0,  0,-10,-20,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-30,  0,  0,  0,  0,-30,-30,
            -50,-30,-30,-30,-30,-30,-30,-50
        ]
        flipped_piece_position_values[chess.KING] = list(reversed(piece_position_values[chess.KING]))
    
    # Evaluate each legal move
    for move in legal_moves:
        score = 0
        from_square = move.from_square
        to_square = move.to_square
        
        # Get the moving piece
        piece = board.piece_at(from_square)
        if not piece:
            continue
            
        # 1. Capture evaluation (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
        if board.is_capture(move):
            victim = board.piece_at(to_square)
            if victim:
                # Calculate capture value
                score += piece_values.get(victim.piece_type, 0) - piece_values.get(piece.piece_type, 0) / 10
                
                # Promotion and capture is very good
                if move.promotion:
                    score += piece_values[move.promotion] - piece_values[chess.PAWN]
        
        # 2. Promotion without capture
        elif move.promotion:
            score += piece_values[move.promotion] - piece_values[chess.PAWN]
            
        # 3. Position evaluation for the moving piece
        position_table = flipped_piece_position_values[piece.piece_type] if piece.color == chess.BLACK else piece_position_values[piece.piece_type]
        # Subtract value of current position
        score -= position_table[from_square] / 10
        # Add value of new position
        score += position_table[to_square] / 10
            
        # 4. Look ahead evaluation
        test_board.push(move)
        
        # Check if we give check
        if test_board.is_check():
            score += 50
            
            # Checkmate is best
            if test_board.is_checkmate():
                score += 10000
                
        # Avoid stalemate
        if test_board.is_stalemate():
            score -= 5000
            
        # Count attacked and defended pieces after our move
        attack_defend_score = 0
        for sq in chess.SQUARES:
            target_piece = test_board.piece_at(sq)
            if not target_piece:
                continue
                
            # Count our attacks on opponent pieces
            if target_piece.color != piece.color and test_board.is_attacked_by(piece.color, sq):
                attack_defend_score += piece_values.get(target_piece.piece_type, 0) / 30
                
            # Penalize leaving our pieces hanging
            if target_piece.color == piece.color and test_board.is_attacked_by(not piece.color, sq):
                if not test_board.is_attacked_by(piece.color, sq):
                    attack_defend_score -= piece_values.get(target_piece.piece_type, 0) / 15
        
        score += attack_defend_score
        
        # Special opening rules
        if game_phase == "opening":
            # Encourage castling
            if piece.piece_type == chess.KING and abs(from_square - to_square) == 2:
                score += 150
                
            # Encourage development of minor pieces
            if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                # Check if the piece starts on its initial square
                init_knight_squares = [chess.B1, chess.G1] if piece.color == chess.WHITE else [chess.B8, chess.G8]
                init_bishop_squares = [chess.C1, chess.F1] if piece.color == chess.WHITE else [chess.C8, chess.F8]
                
                if (piece.piece_type == chess.KNIGHT and from_square in init_knight_squares) or \
                   (piece.piece_type == chess.BISHOP and from_square in init_bishop_squares):
                    score += 50
                    
            # Discourage early queen development
            if piece.piece_type == chess.QUEEN and len(board.move_stack) < 10:
                score -= 50
        
        # Reset the board
        test_board.pop()
        
        # Store move score
        move_scores[move] = score
    
    # Get the best scoring move
    if move_scores:
        best_moves = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)
        best_move = best_moves[0][0]
        logger.info(f"Best engine move: {best_move.uci()} with score {move_scores[best_move]}")
        
        # Add some randomness for lower-scored moves to create variety
        if len(best_moves) > 1 and random.random() < 0.1:  # 10% chance to pick a random top move
            # Choose randomly from top 3 moves
            top_n = min(3, len(best_moves))
            selected_move = best_moves[random.randint(0, top_n-1)][0]
            logger.info(f"Randomly selecting alternative move from top {top_n}: {selected_move.uci()}")
            return selected_move
            
        return best_move
    
    # If no scored moves (shouldn't happen), return first legal move
    return legal_moves[0]

def retry_llama_move(board, legal_moves, previous_suggestion=None):
    """Retry getting a move from Llama with a more explicit prompt."""
    try:
        # Create a simplified board representation
        board_ascii = str(board)
        fen = board.fen()
        # Just list the legal moves in UCI format
        legal_moves_uci = [move.uci() for move in legal_moves]
        
        # More directive prompt focused on choosing from the list
        prompt = f"""You must select a legal chess move.
Board position: {fen}
LEGAL MOVES: {', '.join(legal_moves_uci)}

Select ONE move from the legal moves list above.
Respond with ONLY the move in UCI format (e.g., "e2e4"). Do not add any explanation.
"""
        
        if previous_suggestion:
            prompt += f"\nNOTE: Your previous suggestion '{previous_suggestion}' was not a legal move in this position."
        
        # Call the Llama API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLAMA_API_KEY}"
        }
        data = { 
            "model": "llama-3.3-70b",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a chess engine. Your only task is to output a single legal chess move in UCI notation."
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 10,
            "temperature": 0.0,  # Pure deterministic output
            "top_p": 1.0
        }
        logger.info("Retrying Llama API with simplified prompt")
        response = requests.post(LLAMA_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code != 200:
            raise ValueError(f"Llama API retry error: {response.status_code}")
        
        # Parse the response
        response_data = response.json()
        move_text = response_data["choices"][0]["message"]["content"].strip().lower()
        logger.info(f"Llama retry suggested: {move_text}")
        
        # Extract any UCI pattern
        import re
        move_pattern = re.compile(r'([a-h][1-8][a-h][1-8][qrbnk]?)')
        match = move_pattern.search(move_text)
        
        if match:
            suggested_move = match.group(1)
            for m in legal_moves:
                if m.uci() == suggested_move:
                    return m
        logger.warning("Llama retry failed to suggest a valid move.")
        return None
    except Exception as e:
        logger.error(f"Error in retry_llama_move: {str(e)}")
        return None

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

@app.route('/restart_game', methods=['POST'])
def restart_game():
    data = request.json
    game_id = data.get('game_id')
    mode = data.get('mode', 'regular')
    
    logger.info(f"Restarting game {game_id} in {mode} mode")
    
    # Handle the case where we're restarting a non-existent game
    if game_id == 'new' or game_id not in games:
        game_id = str(random.randint(1000, 9999))
        logger.info(f"Creating new game with ID {game_id} instead of restarting")
    
    games[game_id] = {
        'board': chess.Board(),
        'mode': mode
    }
    
    return jsonify({
        'game_id': game_id,
        'fen': games[game_id]['board'].fen(),
        'mode': mode
    })

if __name__ == '__main__':
    app.run(debug=True)

def save_llama_interaction(prompt, response_text, suggested_move, fen):
    """Save the interaction with Llama API for debugging."""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a filename based on timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(log_dir, f"llama_interaction_{timestamp}.txt")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"====== BOARD POSITION (FEN) ======\n")
        f.write(f"{fen}\n\n")
        f.write(f"====== PROMPT TO LLAMA ======\n")
        f.write(f"{prompt}\n\n")
        f.write(f"====== LLAMA RESPONSE ======\n")
        f.write(f"{response_text}\n\n")
        f.write(f"====== EXTRACTED MOVE ======\n")
        f.write(f"{suggested_move if suggested_move else 'No valid move extracted'}\n")
    
    logger.info(f"Saved Llama interaction to {filename}")
