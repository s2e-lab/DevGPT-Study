import micropip

async def install_opencv():
    await micropip.install("opencv-python")

# Call the async function immediately using an async IIFE
(async () => {
    await install_opencv();
    // Your code that uses opencv-python can go here
})();
