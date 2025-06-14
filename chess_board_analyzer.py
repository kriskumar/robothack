#!/usr/bin/env python3
"""
Simple Chess Board Analyzer - Uses OpenAI GPT-4o-mini and Stockfish
"""

import base64
import os
import requests
import chess
from stockfish import Stockfish
import argparse
import re

class SimpleChessAnalyzer:
    def __init__(self, stockfish_path: str = None, api_key: str = None):
        """
        Initialize the chess analyzer
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize Stockfish
        if stockfish_path and os.path.exists(stockfish_path):
            self.stockfish = Stockfish(path=stockfish_path, depth=15, parameters={"Threads": 2, "Minimum Thinking Time": 30})
        else:
            # Try common Stockfish locations
            for path in ["/Users/kris/bin/stockfish", "/usr/local/bin/stockfish", "/usr/bin/stockfish", "stockfish"]:
                try:
                    self.stockfish = Stockfish(path=path, depth=15, parameters={"Threads": 2, "Minimum Thinking Time": 30})
                    print(f"Using Stockfish at: {path}")
                    break
                except Exception as e:
                    print(f"Failed to initialize Stockfish at {path}: {e}")
                    continue
            else:
                raise Exception("Stockfish not found. Please install Stockfish or provide path.")
    
    def encode_image(self, image_path: str) -> str:
        """Convert image to base64 string for OpenAI API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def get_fen_from_image(self, image_path: str) -> str:
        """
        Use OpenAI GPT-4o-mini to analyze chess board and return FEN notation
        """
        base64_image = self.encode_image(image_path)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = """Analyze this chess board image and return ONLY the board position in FEN notation.

Identify pieces: K/k=King, Q/q=Queen, R/r=Rook, B/b=Bishop, N/n=Knight, P/p=Pawn (White/Black)
Use numbers for consecutive empty squares.

Return ONLY the board part (before the space in FEN), nothing else.
Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR

Do not include any explanatory text, just the position string."""

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['choices'][0]['message']['content'].strip()
                
                # DEBUG: Print raw response from OpenAI
                print(f"\n🔍 RAW OPENAI RESPONSE:")
                print(f"'{response_text}'")
                print(f"🔍 END RAW RESPONSE\n")
                
                # Extract FEN from response
                fen_board = self.extract_fen_from_response(response_text)
                
                print(f"🔧 EXTRACTED FEN BOARD: '{fen_board}'")
                
                if not fen_board or fen_board == "UNCLEAR":
                    return None
                
                # Determine castling rights based on piece positions
                castling_rights = self.determine_castling_rights(fen_board)
                
                # Add game state info to make complete FEN
                complete_fen = f"{fen_board} w {castling_rights} - 0 1"
                
                print(f"🎯 COMPLETE FEN: '{complete_fen}'")
                
                return complete_fen
            else:
                print(f"OpenAI API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None
    
    def determine_castling_rights(self, fen_board: str) -> str:
        """
        Determine castling rights based on piece positions
        """
        ranks = fen_board.split('/')
        
        # Check rank 8 (black back rank) - index 0
        rank8 = ranks[0]
        # Check rank 1 (white back rank) - index 7  
        rank1 = ranks[7]
        
        castling = ""
        
        # White castling rights
        # King must be on e1 and rook on h1 for kingside
        if self.piece_at_position(rank1, 4) == 'K' and self.piece_at_position(rank1, 7) == 'R':
            castling += "K"
        # King must be on e1 and rook on a1 for queenside    
        if self.piece_at_position(rank1, 4) == 'K' and self.piece_at_position(rank1, 0) == 'R':
            castling += "Q"
            
        # Black castling rights
        # King must be on e8 and rook on h8 for kingside
        if self.piece_at_position(rank8, 4) == 'k' and self.piece_at_position(rank8, 7) == 'r':
            castling += "k"
        # King must be on e8 and rook on a8 for queenside
        if self.piece_at_position(rank8, 4) == 'k' and self.piece_at_position(rank8, 0) == 'r':
            castling += "q"
        
        # If no castling rights, return "-"
        return castling if castling else "-"
    
    def piece_at_position(self, rank_str: str, file_index: int) -> str:
        """
        Get the piece at a specific file in a rank string
        Returns the piece character or None if empty
        """
        current_file = 0
        
        for char in rank_str:
            if char.isdigit():
                # Empty squares
                empty_count = int(char)
                if current_file <= file_index < current_file + empty_count:
                    return None  # Empty square
                current_file += empty_count
            else:
                # Piece
                if current_file == file_index:
                    return char
                current_file += 1
        
        return None
    
    def extract_fen_from_response(self, response_text: str) -> str:
        """
        Extract FEN notation from GPT response, handling various formats
        """
        print(f"🔧 EXTRACTING FEN FROM: '{response_text}'")
        
        # Remove backticks and clean up
        text = response_text.strip().replace('`', '')
        print(f"🔧 AFTER CLEANING: '{text}'")
        
        # Remove common prefixes
        patterns_to_remove = [
            r"The FEN notation.*?is:?\s*",
            r"FEN notation:?\s*",
            r"Position:?\s*",
            r"Board position:?\s*",
            r"The position is:?\s*"
        ]
        
        for pattern in patterns_to_remove:
            old_text = text
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            if old_text != text:
                print(f"🔧 REMOVED PATTERN '{pattern}': '{text}'")
        
        # Split by lines and find the line that looks like FEN
        lines = text.split('\n')
        print(f"🔧 SPLIT INTO LINES: {lines}")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if self.looks_like_fen_board(line):
                print(f"🔧 FOUND FEN-LIKE LINE {i}: '{line}'")
                return line
        
        # If no clear FEN found, return the cleaned text
        final_text = text.strip()
        print(f"🔧 NO FEN PATTERN FOUND, RETURNING: '{final_text}'")
        return final_text
    
    def looks_like_fen_board(self, text: str) -> bool:
        """
        Check if text looks like a valid FEN board position
        """
        print(f"🔍 CHECKING IF FEN-LIKE: '{text}'")
        
        # Basic FEN board pattern
        fen_pattern = r'^[rnbqkpRNBQKP12345678/]+$'
        
        if not re.match(fen_pattern, text):
            print(f"🔍 FAILED REGEX PATTERN")
            return False
        
        # Should have exactly 7 '/' characters (8 ranks)
        slash_count = text.count('/')
        if slash_count != 7:
            print(f"🔍 WRONG SLASH COUNT: {slash_count} (need 7)")
            return False
        
        # Each rank should be valid
        ranks = text.split('/')
        for i, rank in enumerate(ranks):
            if not self.is_valid_rank(rank):
                print(f"🔍 INVALID RANK {i}: '{rank}'")
                return False
        
        print(f"🔍 LOOKS LIKE VALID FEN!")
        return True
    
    def is_valid_rank(self, rank: str) -> bool:
        """
        Check if a rank string is valid (sums to 8 squares)
        """
        total = 0
        for char in rank:
            if char.isdigit():
                total += int(char)
            elif char in 'rnbqkpRNBQKP':
                total += 1
            else:
                print(f"🔍 INVALID CHARACTER IN RANK '{rank}': '{char}'")
                return False
        
        if total != 8:
            print(f"🔍 RANK '{rank}' SUMS TO {total} (need 8)")
            return False
            
        return True
    
    def test_stockfish(self) -> bool:
        """Test if Stockfish is working with a simple position"""
        try:
            test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            
            if not self.stockfish.is_fen_valid(test_fen):
                print("Stockfish cannot validate even the starting position")
                return False
                
            self.stockfish.set_fen_position(test_fen)
            best_move = self.stockfish.get_best_move()
            
            if best_move:
                print(f"Stockfish test successful. Best opening move: {best_move}")
                return True
            else:
                print("Stockfish test failed - no best move returned")
                return False
                
        except Exception as e:
            print(f"Stockfish test failed: {e}")
            return False
    
    def print_board_visualization(self, fen: str):
        """Print a visual representation of the board for debugging"""
        try:
            board = chess.Board(fen)
            print(f"🔍 BOARD VISUALIZATION:")
            print(board)
            print()
        except Exception as e:
            print(f"❌ Cannot visualize board: {e}")
    
    def validate_fen(self, fen: str) -> bool:
        """Check if FEN string is valid using python-chess with detailed debugging"""
        try:
            print(f"🔍 VALIDATING FEN: {fen}")
            
            # Try to create the board
            board = chess.Board(fen)
            
            # Check if it's valid
            if board.is_valid():
                print(f"✅ FEN is valid!")
                return True
            else:
                print(f"❌ Board is invalid according to python-chess")
                
                # Let's check what might be wrong
                print(f"🔍 Board status:")
                print(f"   - Is check: {board.is_check()}")
                print(f"   - Is checkmate: {board.is_checkmate()}")
                print(f"   - Is stalemate: {board.is_stalemate()}")
                print(f"   - Has legal moves: {len(list(board.legal_moves))}")
                
                # Check piece counts
                piece_counts = {}
                for piece_type in [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]:
                    for color in [chess.WHITE, chess.BLACK]:
                        piece = chess.Piece(piece_type, color)
                        count = len(board.pieces(piece_type, color))
                        piece_counts[piece.symbol()] = count
                
                print(f"   - Piece counts: {piece_counts}")
                
                # Check if kings are in check illegally
                if board.is_check():
                    print(f"   - Current player in check: {board.turn}")
                
                return False
                
        except Exception as e:
            print(f"❌ FEN validation error: {e}")
            print(f"   This might indicate a malformed FEN string")
            return False
    
    def get_best_move(self, fen: str) -> str:
        """Get best move from Stockfish with error handling"""
        try:
            if not self.stockfish.is_fen_valid(fen):
                print("Stockfish reports FEN as invalid")
                return None
                
            self.stockfish.set_fen_position(fen)
            best_move = self.stockfish.get_best_move()
            return best_move
            
        except Exception as e:
            print(f"Error getting best move: {e}")
            return None
    
    def get_evaluation(self, fen: str) -> dict:
        """Get position evaluation from Stockfish"""
        try:
            if not self.stockfish.is_fen_valid(fen):
                return None
                
            self.stockfish.set_fen_position(fen)
            evaluation = self.stockfish.get_evaluation()
            return evaluation
        except Exception as e:
            print(f"Error getting evaluation: {e}")
            return None
    
    def uci_to_natural_language(self, uci_move: str, fen: str) -> str:
        """
        Convert UCI move notation to natural language description
        """
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci_move)
            
            # Get piece being moved
            moving_piece = board.piece_at(move.from_square)
            if not moving_piece:
                return f"Invalid move: no piece at {chess.square_name(move.from_square)}"
            
            # Get piece names
            piece_names = {
                chess.PAWN: "pawn",
                chess.ROOK: "rook", 
                chess.KNIGHT: "knight",
                chess.BISHOP: "bishop",
                chess.QUEEN: "queen",
                chess.KING: "king"
            }
            
            piece_name = piece_names.get(moving_piece.piece_type, "piece")
            from_square = chess.square_name(move.from_square)
            to_square = chess.square_name(move.to_square)
            
            # Check for special moves
            if board.is_castling(move):
                if move.to_square > move.from_square:
                    return "Castle kingside (move king to g-file and rook to f-file)"
                else:
                    return "Castle queenside (move king to c-file and rook to d-file)"
            
            # Check if it's a capture
            captured_piece = board.piece_at(move.to_square)
            is_capture = captured_piece is not None
            
            # Check for en passant
            if board.is_en_passant(move):
                return f"Move the {piece_name} from {from_square} to {to_square} (en passant capture)"
            
            # Check for pawn promotion
            if move.promotion:
                promotion_names = {
                    chess.QUEEN: "queen",
                    chess.ROOK: "rook", 
                    chess.BISHOP: "bishop",
                    chess.KNIGHT: "knight"
                }
                promotion_piece = promotion_names.get(move.promotion, "queen")
                if is_capture:
                    return f"Move the {piece_name} from {from_square} to {to_square}, capture the piece there, and promote to {promotion_piece}"
                else:
                    return f"Move the {piece_name} from {from_square} to {to_square} and promote to {promotion_piece}"
            
            # Regular moves
            if is_capture:
                captured_name = piece_names.get(captured_piece.piece_type, "piece")
                return f"Move the {piece_name} from {from_square} to {to_square} and capture the {captured_name}"
            else:
                return f"Move the {piece_name} from {from_square} to {to_square}"
                
        except Exception as e:
            return f"Error describing move {uci_move}: {e}"
    
    def get_detailed_move_description(self, uci_move: str, fen: str) -> dict:
        """
        Get detailed information about a move for robot execution
        """
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci_move)
            
            moving_piece = board.piece_at(move.from_square)
            captured_piece = board.piece_at(move.to_square)
            
            return {
                "uci": uci_move,
                "from_square": chess.square_name(move.from_square),
                "to_square": chess.square_name(move.to_square),
                "moving_piece": moving_piece.symbol() if moving_piece else None,
                "captured_piece": captured_piece.symbol() if captured_piece else None,
                "is_capture": captured_piece is not None,
                "is_castling": board.is_castling(move),
                "is_en_passant": board.is_en_passant(move),
                "promotion": chess.piece_name(move.promotion) if move.promotion else None,
                "natural_language": self.uci_to_natural_language(uci_move, fen)
            }
        except Exception as e:
            return {"error": f"Could not analyze move: {e}"}

    def analyze(self, image_path: str) -> dict:
        """
        Complete analysis: image -> FEN -> best move
        """
        # Check if image file exists
        if not os.path.exists(image_path):
            return {"error": "Image file not found"}
        
        print("Analyzing chess board with OpenAI GPT-4o-mini...")
        
        # Test Stockfish first
        if not self.test_stockfish():
            return {"error": "Stockfish is not working properly"}
        
        # Get FEN from image
        fen = self.get_fen_from_image(image_path)
        if not fen:
            return {"error": "Could not analyze board position"}
        
        # Show what the board looks like
        self.print_board_visualization(fen)
        
        # Validate FEN
        if not self.validate_fen(fen):
            return {"error": f"Invalid FEN detected: {fen}"}
        
        print(f"✅ Position validated: {fen}")
        
        # Get best move and evaluation
        best_move = self.get_best_move(fen)
        evaluation = self.get_evaluation(fen)
        
        # Convert UCI move to algebraic notation and natural language
        algebraic_move = None
        move_description = None
        move_details = None
        
        if best_move:
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(best_move)
                algebraic_move = board.san(move)
                move_description = self.uci_to_natural_language(best_move, fen)
                move_details = self.get_detailed_move_description(best_move, fen)
            except Exception as e:
                print(f"Error processing move: {e}")
        
        return {
            "fen": fen,
            "best_move": best_move,
            "algebraic_move": algebraic_move,
            "move_description": move_description,
            "move_details": move_details,
            "evaluation": evaluation
        }

def main():
    parser = argparse.ArgumentParser(description="Simple Chess Board Analyzer")
    parser.add_argument("image", help="Path to chess board image")
    parser.add_argument("--stockfish-path", help="Path to Stockfish binary")
    parser.add_argument("--api-key", help="OpenAI API key")
    
    args = parser.parse_args()
    
    try:
        analyzer = SimpleChessAnalyzer(
            stockfish_path=args.stockfish_path,
            api_key=args.api_key
        )
        
        result = analyzer.analyze(args.image)
        
        print("\n" + "="*50)
        print("CHESS ANALYSIS RESULTS")
        print("="*50)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"📋 Position (FEN): {result['fen']}")
            print(f"🎯 Best Move: {result['best_move']}")
            if result['algebraic_move']:
                print(f"♟️  Move Notation: {result['algebraic_move']}")
            if result['move_description']:
                print(f"🤖 Robot Command: {result['move_description']}")
            if result['move_details']:
                details = result['move_details']
                print(f"📝 Move Details:")
                print(f"   From: {details['from_square']} → To: {details['to_square']}")
                print(f"   Piece: {details['moving_piece']}")
                if details['is_capture']:
                    print(f"   Captures: {details['captured_piece']}")
                if details['is_castling']:
                    print(f"   Special: Castling move")
                if details['promotion']:
                    print(f"   Promotion: {details['promotion']}")
            if result['evaluation']:
                eval_info = result['evaluation']
                if eval_info['type'] == 'cp':
                    centipawns = eval_info['value']
                    advantage = centipawns / 100
                    print(f"📊 Evaluation: {advantage:+.2f} (White advantage)")
                elif eval_info['type'] == 'mate':
                    print(f"🏁 Mate in: {eval_info['value']} moves")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
