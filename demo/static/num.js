
function initWidget() {

    // General-purpose event handler to determine the mouse position relative 
    // to the canvas element
    function standardiseCoords(ev) {
        if (ev.layerX || ev.layerX == 0) { // Firefox
            ev._x = ev.layerX;
            ev._y = ev.layerY;
        } else if (ev.offsetX || ev.offsetX == 0) { // Opera
            ev._x = ev.offsetX;
            ev._y = ev.offsetY;
        }
        return ev;
    }

    var onIVSubmit = function() {

        var data = context.getImageData(
            0, 0, canvas.width, canvas.height).data;

        // Extract alpha values (there is no color)
        var matrix = new Array();
        for (var i = 3, j = 0; i < data.length; i += 4, j++)
            matrix[j] = data[i];
    
        // Reduce image size from 308x308 to 28x28
        var sampleSize = canvas.width / 28;
        var result = Array();
        var ri = 0;

        var get_xy = function(a, w, x, y) {
            return a[(y * w) + x]
        }

        for (var y = 0; y < canvas.height; y += sampleSize)
            for (var x = 0; x < canvas.width; x += sampleSize) {

                var block = Array();
                var i = 0;
                for (var y2 = 0; y2 < sampleSize; y2++)
                    for (var x2 = 0; x2 < sampleSize; x2++) {
                        block[i] = get_xy(matrix, canvas.width, x+x2, y+y2);
                        i++;
                    }

                var sum = 0;
                for (var i = 0; i < block.length; i++)
                    sum += block[i];
                var av = sum / block.length;

                result[ri] = av;
                ri++;
            }

        // To real number between 1 and 0
        for (var i = 0; i < result.length; i++)
            result[i] = result[i] / 255;

        khann.evalIV("num", result, onIVResponse);
    };

    var onIVResponse = function(data) {

        // Pick the digit with the highest output vector score
        var highI, highVal = 0;
        for (var i = 0; i < data.ov_real.length; i++)
            if (data.ov_real[i] > highVal) {
                highVal = data.ov_real[i];
                highI = i;
            }

        // Convert to numeric word
        var word = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine"
            }[highI.toString()];
    
        $("#result").html(word);
    };

    var onResetWidget = function() {
        context.clearRect(0, 0, canvas.width, canvas.height);
    };

    // Get handle to canvas and context
    var canvas, context;
    canvas = document.getElementById('canvas');
    if (!canvas) {
        alert("Can't find canvas element");
        return;
    }
    if (!canvas.getContext) {
        alert("No canvas.getContext!");
        return;
    }
    context = canvas.getContext('2d');
    if (!context) {
        alert("Failed to getContext!");
        return;
    }

    var started = false;
    context.lineWidth = 20;
    context.lineCap = "round";

    // Wire up canvas event handlers for pointer and touch-based devices
    var onMouseDown = function(ev) {
        ev = standardiseCoords(ev);
        context.beginPath();
        context.moveTo(ev._x, ev._y);
        started = true;
    };
    var onMouseMove = function(ev) {
        ev = standardiseCoords(ev);
        if (started) {
            context.lineTo(ev._x, ev._y);
            context.stroke();
        }
    };
    var onMouseUp = function(ev) {
        ev = standardiseCoords(ev);
        if (started)
            started = false;
    };
    canvas.addEventListener("mousedown", onMouseDown, false);
    canvas.addEventListener("mousemove", onMouseMove, false);
    canvas.addEventListener("mouseup", onMouseUp, false);
    canvas.addEventListener("touchstart", onMouseDown, false);
    canvas.addEventListener("touchmove", onMouseMove, false);
    canvas.addEventListener("touchend", onMouseUp, false);

    // Prevent elastic scrolling on touch-based device
    document.body.addEventListener('touchmove', function(event) {
        event.preventDefault();
    },  false); 

    khann.registerCallbacks(onResetWidget, null);
    $("#youDrew").hide();
    $("#ivSubmit").click(onIVSubmit);

}

