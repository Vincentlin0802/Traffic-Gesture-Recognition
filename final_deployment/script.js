const videoElement = document.getElementById('videoElement');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const button = document.getElementById('startButton');
var shouldSendRequests = true;
var intervalId = null;


// Function to start the camera and stream the video to the video element.
async function startCamera() {
    try {
        // Request the media devices for a video stream
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
    } catch (err) {
        console.error('Error accessing the camera:', err);
    }
}

// Function to display a countdown GIF and play a sound.
function showGif() {
    // Get the GIF container and audio elements.
    const gifContainer = document.getElementById('countdownGif');
    const countdownAudio = document.getElementById('countdownAudio');

    // Hide the start button while the countdown is displayed.
    button.classList.add('hidden');
    document.getElementById('resultContainer').innerText = "";

    // Refresh the GIF source to restart the animation.
    const originalSrc = gifContainer.src.split('?')[0];
    gifContainer.src = `${originalSrc}?${new Date().getTime()}`;

    // Show the countdown GIF and play the countdown sound.
    gifContainer.classList.remove('hidden');
    countdownAudio.play();

    // Hide the GIF after 3 seconds.
    setTimeout(() => {
        gifContainer.classList.add('hidden');
    }, 3000);
}


// Function to start sending frames to the server for processing.
function startSendingFrames() {
    console.log("Start");
    shouldSendRequests = true;
    showGif();
     // Wait for the countdown to finish before starting to send frames.
    setTimeout(() => {
        console.log("Starting to send frames");
        intervalId = setInterval(sendFrame, 100);
    }, 3500);
}

// Function to display the result of gesture recognition.
function displayResult(result) {
    var message = "";
    switch(result) {
        case 0:
            message = "NO GESTURE";
            break;
        case 1:
            message = "STOP";
            break;
        case 2:
            message = "MOVE STRAIGHT";
            break;
        case 3:
            message = "LEFT TURN";
            break;
        case 4:
            message = "LEFT TURN WAITING";
            break;
        case 5:
            message = "RIGHT TURN";
            break;
        case 6:
            message = "LANE CHANGING";
            break;
        case 7:
            message = "SLOW DOWN";
            break;
        case 8:
            message = "PULL OVER";
            break;
        default:
            message = "Unknown result";
        }
    document.getElementById('resultContainer').innerText = message;
}

// Function to capture a frame from the video and send it to the back- end.
function sendFrame() {
    if (!shouldSendRequests) {
        console.log("Stopped sending requests.");
        if (intervalId !== null) {
            clearInterval(intervalId);
            intervalId = null;
        }
        return;
    }

    const videoWidth = videoElement.videoWidth;
    const videoHeight = videoElement.videoHeight;

    // Draw the current frame from the video element to the canvas.
    context.drawImage(videoElement, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('frame', blob);
        
        // Send an asynchronous POST request to the server with the frame.
        $.ajax({
            type: 'POST',
            url: 'http://127.0.0.1:5000/predict',
            data: formData,
            processData: false,
            contentType: false,
            success: function(result) {
                console.log(result.result);
                // Check if the result is a number between 0 and 8.
                if (typeof result.result === 'number' && result.result>=0 && result.result <=8) {
                    displayResult(result.result);
                    button.classList.remove('hidden');

                    // Stop sending requests since we got a valid result.
                    shouldSendRequests = false;
                    console.log("Received a number between 0 and 8, stopping requests.");
                }
            },
            error: function() {
                console.log("Fail to post the data");
            }
        });
    }, 'image/jpeg');
}