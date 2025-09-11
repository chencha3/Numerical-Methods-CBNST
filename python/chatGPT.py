import argparse
import os
import re
import numpy as np
from typing import List, Dict
import json
import openai
import time
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY")
Model = "gpt-4.1"

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

class ChatGPT: 
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.0):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.model_version = self._get_model_version(model)

    def _get_model_version(self, model: str) -> str:
        """Extract version from model name for result formatting."""
        if "gpt-4" in model.lower():
            if "turbo" in model.lower():
                return "4-Turbo"
            elif "o1" in model.lower():
                return "4o1"
            else:
                return "4"
        elif "gpt-3.5" in model.lower():
            return "3.5-Turbo"
        else:
            return model.replace("-", "").replace("_", "")    

    def extract_prediction_range(self, response_text: str) -> tuple[int, int]:
        """Extract the predicted iteration range from ChatGPT response."""
        
        # Look for patterns like [min, max] or "min to max" or "min-max"
        patterns = [
            r'\[(\d+),\s*(\d+)\]',  # [min, max]
            r'\[(\d+)-(\d+)\]',     # [min-max]
            r'(\d+)\s*to\s*(\d+)',  # min to max
            r'(\d+)\s*-\s*(\d+)',   # min - max
            r'between\s*(\d+)\s*and\s*(\d+)',  # between min and max
            r'from\s*(\d+)\s*to\s*(\d+)',      # from min to max
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                min_val = int(match.group(1))
                max_val = int(match.group(2))
                return min_val, max_val
        
        # If no range found, look for single number and create a range
        single_num_pattern = r'(\d+)\s*iterations?'
        match = re.search(single_num_pattern, response_text, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            # Create a range around the single number (±20%)
            margin = max(1, int(num * 0.2))
            return num - margin, num + margin
        
        # Default fallback
        return 0, 0

    def create_prediction_prompt(self, matrix_info: Dict) -> str:
        """Create a prompt for matrix iteration prediction.""" 

        prompt_template = """
You are an expert in numerical analysis and iterative methods. I will provide you with an matrix for an iterative solver.

Matrix Information:
- Dimensions: {dimensions}

Matrix Data:
{matrix}

Based on this matrix, please predict the number of iterations Gauss Seidel would require to converge. Consider:

1. The diagonal dominance of the matrix
2. The condition number and eigenvalue distribution
3. The magnitude and distribution of off-diagonal elements
4. Typical convergence patterns for similar systems
5. The convergence tolerance is 0.000000001

Please provide your prediction as a range in the format: "Predicted iterations: [min, max]"
where min and max are integer values representing the likely range of iterations needed.
"""
        
        # Get matrix sample (first 5x5 for prompt efficiency)
        matrix = np.array(matrix_info['matrix_array'])
        
        # Format matrix sample
        matrix_str = ""
        for row in matrix:
            row_str = " ".join([f"{val:10.5f}" for val in row])
            matrix_str += row_str + "\n"
        
        prompt = prompt_template.format(
            dimensions=matrix_info['dimensions'],
            matrix=matrix_str.strip()
        )
        
        return prompt
    

    def predict(self, matrix_info: Dict) -> Dict:
        """Send matrix to ChatGPT and get iteration prediction.""" 
        prompt = self.create_prediction_prompt(matrix_info)
        delay = 1.0 
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert in numerical analysis, linear algebra, and iterative methods for solving linear systems."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=800,
                temperature=0.1  # Lower temperature for more consistent predictions
            )
            
            prediction_text = response.choices[0].message.content.strip()
            
            # Extract predicted range
            min_pred, max_pred = self.extract_prediction_range(prediction_text)
            
            # Format result key
            result_key = f"GPT-{self.model_version} Predicted Iterations in range: [{min_pred}, {max_pred}]" 
            return {
                'success': True,
                'prediction_text': prediction_text,
                'predicted_range': [min_pred, max_pred],
                'result_key': result_key,
                'actual_iterations': matrix_info['iterations'],
                'matrix_info': {
                    'file_name': matrix_info['file_name'],
                    'matrix_index': matrix_info['matrix_index'],
                    'dimensions': matrix_info['dimensions']
                },
                'model_used': self.model
            }

        except openai.RateLimitError:
            time.sleep(delay)
            delay = min(delay * 2, 20)  # cap the backoff 
        except Exception as e:
            print(f"    Error during prediction: {e}")
            return {
                'success': False,
                'error': str(e),
                'matrix_info': matrix_info,
                'model_used': self.model
            }

class MatrixParser:
    """Class to parse matrices from text files and save them."""
    
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
    
    def parse_matrices_from_file(self, file_path: str) -> List[Dict]:
        """Parse all matrices from a single file."""
        matrices = []
        
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Find all matrix headers with dimensions
        header_pattern = r'The Augmented Matrix \((\d+x\d+)\)'
        headers = list(re.finditer(header_pattern, content))
        
        for i, header_match in enumerate(headers):
            # Get the dimensions from the header
            dimensions = header_match.group(1)
            
            # Find the start and end of this matrix block
            start_pos = header_match.end()
            
            if i < len(headers) - 1:
                # Not the last matrix, end before next header
                end_pos = headers[i + 1].start()
            else:
                # Last matrix, go to end of content
                end_pos = len(content)
            
            # Extract the matrix block
            matrix_block = content[start_pos:end_pos].strip()
            
            # Parse this matrix
            matrix_data = self._parse_single_matrix_block(matrix_block, dimensions)
            
            if matrix_data is not None:
                matrix_info = {
                    'file_name': os.path.basename(file_path),
                    'matrix_index': i + 1,
                    'dimensions': dimensions,
                    'matrix': matrix_data['matrix'].tolist(),  # Convert to list for JSON serialization
                    'matrix_array': matrix_data['matrix'],  # Keep numpy array for processing
                    'iterations': matrix_data['iterations']
                }
                matrices.append(matrix_info)
        
        return matrices
    
    def _parse_single_matrix_block(self, block: str, dimensions: str) -> Dict:
        """Parse a single matrix block."""
        lines = block.strip().split('\n')
        
        # Find iteration count
        iterations = None
        matrix_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('Tot number of iterations:'):
                iterations = int(line.split(':')[1].strip())
            elif line and not line.startswith('Tot number') and line:
                # This is a matrix row - check if it contains numbers
                try:
                    # Try to parse the first element to see if it's a number
                    test_elements = line.split()
                    if test_elements and self._is_number(test_elements[0]):
                        matrix_lines.append(line)
                except:
                    continue
        
        if not matrix_lines:
            return None
        
        # Parse matrix data
        matrix_data = []
        for line in matrix_lines:
            # Split by whitespace and convert to float
            row = []
            for element in line.split():
                try:
                    row.append(float(element))
                except ValueError:
                    continue
            if row:  # Only add non-empty rows
                matrix_data.append(row)
        
        if not matrix_data:
            return None
            
        matrix = np.array(matrix_data)
        
        return {
            'matrix': matrix,
            'iterations': iterations
        }
    
    def _is_number(self, s: str) -> bool:
        """Check if a string represents a number."""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _check_accuracy(self, predicted_range: List[int], actual: int) -> Dict:
        """Check if the actual value falls within predicted range."""
        min_pred, max_pred = predicted_range
        within_range = min_pred <= actual <= max_pred
        
        return {
            'within_range': within_range,
            'difference_from_range': 0 if within_range else min(abs(actual - min_pred), abs(actual - max_pred))
        }

    def save_matrices_to_file(self, matrices: List[Dict], output_file_path: str):
        """Save matrices to a new file as formatted text file."""
        with open(output_file_path, 'w') as f:
            for i, matrix_info in enumerate(matrices):
                f.write(f"Matrix {matrix_info['matrix_index']} from {matrix_info['file_name']}\n")
                f.write(f"Dimensions: {matrix_info['dimensions']}\n")
                f.write(f"Iterations: {matrix_info['iterations']}\n")
                f.write(f"{matrix_info['predicted_range']}\n")
                f.write("-" * 50 + "\n")
                
                # Write matrix data
                matrix = matrix_info['matrix_array']
                for row in matrix:
                    row_str = " ".join([f"{val:10.5f}" for val in row])
                    f.write(row_str + "\n")
                
                f.write("=" * 80 + "\n\n")

    def predict(self, matrices: List[Dict], chatgpt: ChatGPT) -> List[Dict]:
        """Predict iterations for a list of matrices using ChatGPT."""
        results = []
        
        for matrix_info in matrices:
            prediction_result = chatgpt.predict(matrix_info)
            predicted_range = prediction_result.get('predicted_range', [0, 0])
            predicted_accuracy = self._check_accuracy(predicted_range, matrix_info['iterations'])
            matrix_info['predicted_range'] = prediction_result.get('result_key', "Failed")
            matrix_info['prediction_accuracy'] = predicted_accuracy.get('within_range', False) 
            results.append(matrix_info) 
        return results
     
    def process_all_files(self):
        """Process all .txt files in the folder and save parsed matrices."""
        
        if not os.path.exists(self.folder_path):
            print(f"Error: Folder '{self.folder_path}' does not exist.")
            return
        
        txt_files = [f for f in os.listdir(self.folder_path) if f.endswith('.txt') and 'predicted' not in f]
        txt_files = sorted(txt_files, key=natural_key)

        
        if not txt_files:
            print(f"No .txt files found in '{self.folder_path}'.")
            return
        
        print(f"Found {len(txt_files)} .txt files to process.")
        
        for filename in txt_files:
            file_path = os.path.join(self.folder_path, filename)
            print(f"\nProcessing: {filename}")
            
            # Parse matrices from file
            matrices = self.parse_matrices_from_file(file_path)
            
            if matrices:
                print(f"  Found {len(matrices)} matrices")

                matrices = self.predict(matrices, chatgpt=ChatGPT(api_key=API_KEY, model=Model))
                
                # Create output filename
                base_name = filename.rsplit('.', 1)[0]  # Remove .txt extension
                output_filename = f"{base_name}_predicted.txt"
                
                output_path = os.path.join(self.folder_path, output_filename)
                
                # Save matrices
                self.save_matrices_to_file(matrices, output_path)
                print(f"  Saved to: {output_filename}")
                
                # Print matrix summary
                for matrix_info in matrices:
                    print(f"    Matrix {matrix_info['matrix_index']}: {matrix_info['dimensions']}, {matrix_info['iterations']} iterations")
            else:
                print(f"  No matrices found in {filename}")

def main():
    """Main function to parse and save matrices."""

    ap = argparse.ArgumentParser(description="Batch matrices -> OpenAI prediction")
    ap.add_argument("folder", help="Folder containing .txt files (each with a matrix).")
    
    # Configuration
    FOLDER_PATH = ap.parse_args().folder
    if not FOLDER_PATH:
        FOLDER_PATH = "."  # Current directory if nothing entered
    
    print(f"\nUsing folder: {FOLDER_PATH}")
    
    # Initialize parser
    parser = MatrixParser(FOLDER_PATH)
    
    # Process all files
    parser.process_all_files()
    
    print("\n✓ Processing complete!")

if __name__ == "__main__":
    if not API_KEY:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        exit(1)
    main()