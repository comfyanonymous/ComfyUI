import os

def create_symbolic_links(src_dir, dst_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        # Compute the relative path from the source directory
        rel_path = os.path.relpath(root, src_dir)
        dest_root = os.path.join(dst_dir, rel_path)
        
        # Ensure the corresponding destination directory exists
        if not os.path.exists(dest_root):
            os.makedirs(dest_root)
        
        # Create symbolic links for the files
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_root, file)
            
            if os.path.exists(dest_file):
                os.remove(dest_file)
                
            os.symlink(src_file, dest_file)
            print(f"Created symbolic link: {dest_file} -> {src_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    args = parser.parse_args()

    # Example usage
    src_directory = args.src
    dst_directory = './models'
    create_symbolic_links(src_directory, dst_directory)
