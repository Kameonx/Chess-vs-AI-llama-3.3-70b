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

# Add OpenAI API key
OPENAI_API_KEY = "sk-" # Add your OpenAI API key here
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

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
    """Get a move exclusively from Llama."""
    return get_llama_move(board)

def get_llama_move(board):
    """Get a stronger move from the Llama 3.3 70B AI model with advanced chess logic."""
    try:
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Just return first move if only one is available
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Create board representation for better context
        board_ascii = str(board)
        fen = board.fen()
        
        # Enhanced prompt with positional understanding and competitive chess concepts
        prompt = f"""Analyze this chess position as a 2800+ rated grandmaster:

{board_ascii}

FEN: {fen}
Turn: {'White' if board.turn == chess.WHITE else 'Black'}

Identify the single best move considering:
1. Material advantage
2. King safety
3. Control of center squares
4. Piece activity and coordination
5. Tactical opportunities (captures, checks, threats)
6. Pawn structure integrity
7. Long-term strategic advantages

Respond with ONLY the best move in UCI format (e.g., "e2e4").
"""
        
        # Call the Llama API with optimized parameters for better output
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLAMA_API_KEY}"
        }
        
        data = {
            "model": "llama-3.3-70b",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a world champion chess grandmaster with exceptional tactical vision. When analyzing a position, you see multiple moves ahead and find the objectively strongest continuation. Respond with only the single best move in UCI notation."
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 10,
            "temperature": 0.0,
            "top_p": 1.0
        }
        
        # Try to get a move from Llama
        logger.info(f"Calling Llama API for chess analysis")
        response = requests.post(LLAMA_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code != 200:
            raise ValueError(f"Llama API error: {response.status_code}")
        
        # Parse the response
        response_data = response.json()
        move_text = response_data["choices"][0]["message"]["content"].strip().lower()
        logger.info(f"Llama suggested: {move_text}")
        
        # Extract the move from the text
        import re
        move_pattern = re.compile(r'([a-h][1-8][a-h][1-8][qrbnk]?)')
        match = move_pattern.search(move_text)
        
        if match:
            suggested_move = match.group(1)
            # Validate the move is legal
            for move in legal_moves:
                if move.uci() == suggested_move:
                    logger.info(f"Using Llama's move: {suggested_move}")
                    return move
            
            logger.warning(f"Llama suggested illegal move: {suggested_move}")
        else:
            logger.warning(f"Could not extract a valid move from: {move_text}")
        
        # If we couldn't get a valid move from Llama, use enhanced evaluation
        return evaluate_best_move(board, legal_moves)
        
    except Exception as e:
        logger.error(f"Error in get_llama_move: {str(e)}")
        return evaluate_best_move(board, legal_moves)

def evaluate_best_move(board, legal_moves):
    """Enhanced move evaluation using more sophisticated chess strategy."""
    if not legal_moves:
        return None
    
    # Copy board for evaluation
    test_board = board.copy()
    
    # Dictionary to store move scores
    move_scores = {}
    
    # Material values
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    # Positional bonus
    # Higher values for central squares for better position control
    positional_values = [
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0, 10, 10, 10, 10,  0,  0,
        0,  0, 10, 20, 20, 10,  0,  0,
        0,  0, 10, 20, 20, 10,  0,  0,
        0,  0, 10, 10, 10, 10,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    
    # Evaluate each move
    for move in legal_moves:
        score = 0
        from_square = move.from_square
        to_square = move.to_square
        
        # Get the piece making the move
        piece = board.piece_at(from_square)
        if not piece:
            continue
        
        # 1. Material evaluation - prioritize captures
        if board.is_capture(move):
            captured_piece = board.piece_at(to_square)
            if captured_piece:
                # Calculate material gain/loss
                attacker_value = piece_values.get(piece.piece_type, 0)
                captured_value = piece_values.get(captured_piece.piece_type, 0)
                score += captured_value - attacker_value/10
        
        # 2. Positional bonus - central control is good
        score += positional_values[to_square]
        
        # 3. Look ahead evaluation - simulate the move
        test_board.push(move)
        
        # 3a. Evaluate check
        if test_board.is_check():
            score += 50  # Giving check is good
            
            # 3b. Evaluate checkmate
            if test_board.is_checkmate():
                score += 10000  # Checkmate is best
        
        # 3c. Count attacked pieces after the move
        attacked_pieces_value = 0
        defended_pieces_value = 0
        for sq in chess.SQUARES:
            attacked_piece = test_board.piece_at(sq)
            if attacked_piece and attacked_piece.color != piece.color and test_board.is_attacked_by(piece.color, sq):
                attacked_pieces_value += piece_values.get(attacked_piece.piece_type, 0) / 10
            elif attacked_piece and attacked_piece.color == piece.color and test_board.is_attacked_by(piece.color, sq):
                defended_pieces_value += piece_values.get(attacked_piece.piece_type, 0) / 20
        
        score += attacked_pieces_value + defended_pieces_value
        
        # 3d. Penalize moves that leave pieces hanging
        piece_at_risk = False
        for sq in chess.SQUARES:
            risk_piece = test_board.piece_at(sq)
            if risk_piece and risk_piece.color == piece.color and test_board.is_attacked_by(not piece.color, sq):
                if not test_board.is_attacked_by(piece.color, sq):
                    # Undefended piece under attack
                    piece_at_risk = True
                    score -= piece_values.get(risk_piece.piece_type, 0) / 5
        
        # Development bonuses
        if len(board.move_stack) < 10:  # early game
            # Encourage knight and bishop development
            if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                # Initial positions of pieces
                knight_starting = [chess.B1, chess.G1, chess.B8, chess.G8]
                bishop_starting = [chess.C1, chess.F1, chess.C8, chess.F8]
                
                # If moving from starting square, it's development
                if (piece.piece_type == chess.KNIGHT and from_square in knight_starting) or \
                   (piece.piece_type == chess.BISHOP and from_square in bishop_starting):
                    score += 30
            
            # Encourage castling
            if piece.piece_type == chess.KING and abs(from_square - to_square) == 2:
                score += 60
            
            # Penalize early queen development
            if piece.piece_type == chess.QUEEN and len(board.move_stack) < 6:
                score -= 20
        
        # Reset the test board
        test_board.pop()
        
        # Store the score
        move_scores[move] = score
    
    # Return the move with the highest score
    if move_scores:
        best_move = max(move_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Advanced evaluation chose move {best_move.uci()} with score {move_scores[best_move]}")
        return best_move
    
    # Fallback to first legal move
    return legal_moves[0]

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
