"""
Demo script showing how to use the DocumentScanner class
"""
from document_scanner import DocumentScanner
import os

def demo_scan():
    """Demonstrate the document scanner functionality"""
    scanner = DocumentScanner()
    
    # Example image path (you'll need to provide your own image)
    sample_images = [
        "sample_document.jpg",
        "receipt.png", 
        "paper.jpeg"
    ]
    
    print("🔍 Looking for sample images...")
    
    # Find the first available sample image
    input_image = None
    for img in sample_images:
        if os.path.exists(img):
            input_image = img
            break
    
    if input_image is None:
        print("📸 No sample images found. Please add an image file to test with.")
        print("Supported formats: .jpg, .jpeg, .png, .bmp")
        return
    
    print(f"📱 Processing: {input_image}")
    
    # Create output directory
    output_dir = "scanned_documents"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate preview first
    print("📋 Generating detection preview...")
    scanner.preview_detection(input_image, f"{output_dir}/detection_preview.png")
    
    # Process with different enhancement modes
    enhancement_modes = ['adaptive', 'otsu', 'simple']
    
    for mode in enhancement_modes:
        print(f"\n🌟 Processing with {mode} enhancement...")
        
        # Create mode-specific output directory
        mode_output = os.path.join(output_dir, mode)
        
        result = scanner.process_document(
            input_image,
            mode_output,
            enhancement_mode=mode,
            save_format='both'
        )
        
        if result is not None:
            print(f"✅ {mode.capitalize()} processing completed!")
        else:
            print(f"❌ {mode.capitalize()} processing failed!")
    
    print(f"\n🎉 Demo completed! Check the '{output_dir}' folder for results.")
    print("📁 You should see:")
    print("  - detection_preview.png (shows detected document edges)")
    print("  - adaptive/ folder with scanned results")
    print("  - otsu/ folder with scanned results") 
    print("  - simple/ folder with scanned results")

if __name__ == "__main__":
    demo_scan()


