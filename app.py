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
        
        # Enhanced prompt with explicit instruction to choose a move
        prompt = f"""As a chess grandmaster, analyze this position:
Board: 
{board_ascii}
FEN: {fen}
Turn: {'White' if board.turn == chess.WHITE else 'Black'}
Legal Moves: {', '.join(legal_moves_descriptions)}

Analyze this position and determine the best move. Consider:
1. Material balance and potential captures
2. Piece development and king safety
3. Center control and space advantage
4. Tactical opportunities (forks, pins, discoveries)
5. Pawn structure and weaknesses

Respond with ONLY the best move in UCI format (e.g., "e2e4").
"""
        
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
                    "content": "You are a world-champion chess engine. Evaluate positions deeply (at least 8 moves ahead). Never hang pieces. Provide the BEST single move in UCI notation."
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 10,
            "temperature": 0.0,  # Make it fully deterministic
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
        return None

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
