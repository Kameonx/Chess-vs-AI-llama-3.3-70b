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
    """Get a move exclusively from Llama."""
    return get_llama_move(board)

def get_llama_move(board):
    """Get a move from the Llama 3.3 70B AI model."""
    try:
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Just return first move if only one is available
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Create board representation for better context
        fen = board.fen()
        logger.info(f"Using board FEN for Llama: {fen}")
        board_ascii = str(board)
        
        # Convert legal moves to a readable format for the model
        legal_moves_descriptions = []
        for move in legal_moves:
            piece = board.piece_at(move.from_square)
            piece_type = piece.symbol().upper() if piece.color == chess.WHITE else piece.symbol().lower()
            capture = "x" if board.is_capture(move) else "-"
            target_piece = board.piece_at(move.to_square)
            target_desc = target_piece.symbol() if target_piece else "empty"
            legal_moves_descriptions.append(f"{move.uci()} ({piece_type} {capture} {target_desc})")
        
        # Track move history to help avoid repetitions
        move_history = [move.uci() for move in board.move_stack]
        repeated_patterns = find_repeated_patterns(move_history)
        
        # Enhanced prompt with clear instructions for the Llama model
        prompt = f"""As a chess grandmaster, analyze this position and choose your next move:

Board:
{board_ascii}

FEN: {fen}
Turn: {'White' if board.turn == chess.WHITE else 'Black'}

Legal Moves (you must pick only from these):
{', '.join(legal_moves_descriptions)}

Previous moves: {' '.join(move_history) if move_history else 'None'}
Note: Avoid repeating moves or positions. Patterns to avoid: {repeated_patterns}

Follow these principles for selecting your move:
1. Material advantage - capture valuable pieces when safe
2. Piece development and king safety
3. Center control and mobility
4. Tactical opportunities (forks, pins, skewers)
5. Forward planning (think multiple moves ahead)

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
                    "content": "You are a chess grandmaster using the Llama 3.3 70B model. Analyze positions carefully and respond with only your chosen move in UCI notation."
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 20,
            "temperature": 0.2,  # Low but non-zero for some creativity
            "top_p": 0.95
        }
        
        # Make the API call
        logger.info(f"Calling Llama API for chess analysis")
        response = requests.post(LLAMA_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code != 200:
            raise ValueError(f"Llama API error: {response.status_code}")
        
        # Parse the response
        response_data = response.json()
        move_text = response_data["choices"][0]["message"]["content"].strip().lower()
        logger.info(f"Llama suggested: {move_text}")
        
        # Extract the UCI move from the text
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
            
            # If move is not legal, retry with a more explicit prompt
            logger.warning(f"Llama suggested illegal move: {suggested_move}, retrying.")
            return retry_llama_move(board, legal_moves, previous_suggestion=suggested_move)
        else:
            logger.warning(f"Could not extract a valid move from: {move_text}")
            return retry_llama_move(board, legal_moves)
        
    except Exception as e:
        logger.error(f"Error in get_llama_move: {str(e)}")
        # If all else fails, use a fallback method but only as a last resort
        return fallback_random_move(legal_moves)

def retry_llama_move(board, legal_moves, previous_suggestion=None):
    """Retry getting a move from Llama with a more explicit prompt."""
    try:
        # Create a simpler prompt focused solely on selecting a legal move
        fen = board.fen()
        legal_moves_uci = [move.uci() for move in legal_moves]
        
        prompt = f"""You are a chess engine. Select the best move from ONLY these legal options:
{', '.join(legal_moves_uci)}

Position FEN: {fen}

Your task is to pick exactly ONE move from the list above.
Respond with ONLY the UCI notation (e.g., "e2e4") and nothing else.
"""
        
        if previous_suggestion:
            prompt += f"\nNOTE: Your previous suggestion '{previous_suggestion}' was not valid."
        
        # Call the Llama API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLAMA_API_KEY}"
        }
        
        data = {
            "model": "llama-3.3-70b",
            "messages": [
                {"role": "system", "content": "You are a chess engine that only outputs valid chess moves in UCI notation."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 10,
            "temperature": 0.0,  # Pure deterministic for retry
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
        
        # Extract move directly or with regex if needed
        if move_text in [move.uci() for move in legal_moves]:
            # Direct match to a legal move
            for move in legal_moves:
                if move.uci() == move_text:
                    return move
        else:
            # Try to extract with regex
            import re
            move_pattern = re.compile(r'([a-h][1-8][a-h][1-8][qrbnk]?)')
            match = move_pattern.search(move_text)
            
            if match:
                suggested_move = match.group(1)
                for move in legal_moves:
                    if move.uci() == suggested_move:
                        return move
        
        # If still no valid move, use fallback
        logger.warning("Llama retry failed to suggest a valid move.")
        return fallback_random_move(legal_moves)
    
    except Exception as e:
        logger.error(f"Error in retry_llama_move: {str(e)}")
        return fallback_random_move(legal_moves)

def fallback_random_move(legal_moves):
    """Last resort fallback: select a random legal move."""
    logger.warning("Using fallback random move selection")
    return random.choice(legal_moves) if legal_moves else None

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
