public class Main {
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK) {
            Bitmap photo = (Bitmap) data.getExtras().get("data");

            // Save the image to a file
            File imageFile = saveImageToFile(photo);

            // Get the file path
            imagePath = imageFile.getAbsolutePath();

            // Display the captured photo in the ImageView
            imageView.setImageBitmap(photo);
        }
    }
}