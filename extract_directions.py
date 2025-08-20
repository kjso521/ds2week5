import re
from pathlib import Path

def extract_unique_directions(data_dir):
    """
    Scans a directory of .npy files and extracts the unique convolution directions
    from the filenames.
    """
    v2_dir = Path(data_dir)
    if not v2_dir.exists():
        print(f"Error: Directory not found at {v2_dir}")
        return []

    unique_directions = set()
    
    # Regex to capture the direction vectors
    pattern = re.compile(r'_dir_(-?\d+\.\d+)_(-?\d+\.\d+)_noise_lv\d+\.npy$')

    for file_path in v2_dir.glob('*.npy'):
        match = pattern.search(file_path.name)
        if match:
            # Create a tuple of floats for the direction vector
            direction = (float(match.group(1)), float(match.group(2)))
            unique_directions.add(direction)
            
    return sorted(list(unique_directions))

def main():
    """
    Main function to extract and print unique directions.
    """
    test_y_v2_path = Path(__file__).parent / "dataset" / "test_y_v2"
    directions = extract_unique_directions(test_y_v2_path)
    
    if directions:
        print(f"Found {len(directions)} unique convolution directions in '{test_y_v2_path.name}':")
        for i, direction in enumerate(directions):
            print(f"  {i+1:2d}: ({direction[0]:.4f}, {direction[1]:.4f})")
    else:
        print("No valid direction vectors found.")

if __name__ == '__main__':
    main()
