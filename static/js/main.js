// Grab elements, create settings, etc.
var video = document.getElementById('video');

// Get access to the camera!
if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    // Not adding `{ audio: true }` since we only want video now
    navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
        //video.src = window.URL.createObjectURL(stream);
        video.srcObject = stream;
        video.play();
    });
}

/* Legacy code below: getUserMedia
else if(navigator.getUserMedia) { // Standard
    navigator.getUserMedia({ video: true }, function(stream) {
        video.src = stream;
        video.play();
    }, errBack);
} else if(navigator.webkitGetUserMedia) { // WebKit-prefixed
    navigator.webkitGetUserMedia({ video: true }, function(stream){
        video.src = window.webkitURL.createObjectURL(stream);
        video.play();
    }, errBack);
} else if(navigator.mozGetUserMedia) { // Mozilla-prefixed
    navigator.mozGetUserMedia({ video: true }, function(stream){
        video.srcObject = stream;
        video.play();
    }, errBack);
}
*/
// Elements for taking the snapshot
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
var video = document.getElementById('video');

// Trigger photo take
document.getElementById("snap").addEventListener("click", function() {
    context.drawImage(video, 0, 0, 640, 480);

	//Trigger image to flask
	let dataURL = canvas.toDataURL('image/png');
	let image = dataURL.split(",")[1];

    var data = JSON.stringify({image})

    var xhr = new XMLHttpRequest();
    xhr.withCredentials = true;

    xhr.addEventListener("readystatechange", function() {
    if(this.readyState === 4) {
        result = this.responseText;
        //console.log(this.responseText);
        var json = JSON.parse(result);
        var canvas1 = document.getElementById('canvas1');
        var ctx = canvas1.getContext('2d');
        var res_image = new Image();
        res_image.src = 'data:image/png;base64,'+json.data.image;
        res_image.onload = function(){
            ctx.drawImage(res_image,0,0,640,480);
        }
    }
    });

    xhr.open("POST", "http://127.0.0.1:5000/api/face");
    xhr.setRequestHeader("Content-Type", "application/json");

    xhr.send(data);
});
