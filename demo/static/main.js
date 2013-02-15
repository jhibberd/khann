
if (window.addEventListener) {

	window.addEventListener('load', function() {
  		var canvas, context, tool;

  		function init () {

    		canvas = document.getElementById('imageView');
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

    		tool = new tool_pencil();

            // Events for pointer-based devices
    		canvas.addEventListener('mousedown', ev_canvas, false);
    		canvas.addEventListener('mousemove', ev_canvas, false);
    		canvas.addEventListener('mouseup', ev_canvas, false);

            // Events for touch-based devices
    		canvas.addEventListener('touchstart', ev_canvas, false);
    		canvas.addEventListener('touchmove', ev_canvas, false);
    		canvas.addEventListener('touchend', ev_canvas, false);

            $("#tryAgain").click(clearImage);

            // Prevent elastic scrolling on touch-based device
            document.body.addEventListener('touchmove', function(event) {
                event.preventDefault();
            },  false); 
  		}

  		function tool_pencil () {

    		var tool = this;
    		this.started = false;
            this.evalTimer = null;

    		context.lineWidth = 20;
    		context.lineCap = "round";

    		this.mousedown = function(ev) {
        		context.beginPath();
        		context.moveTo(ev._x, ev._y);
        		tool.started = true;
                if (tool.evalTimer) {
                    clearTimeout(tool.evalTimer);
                    tool.evalTimer = null;
                }
    		};
            this.touchstart = this.mousedown;

    		this.mousemove = function(ev) {
      			if (tool.started) {
        			context.lineTo(ev._x, ev._y);
        			context.stroke();
      			}
    		};
            this.touchmove = this.mousemove;

    		this.mouseup = function(ev) {
      			if (tool.started) {
        			tool.started = false;
                    tool.evalTimer = setTimeout(evalImage, 2000);
      			}
    		};
            this.touchend = this.mouseup;
  		}

        // General-purpose event handler to determine the mouse position 
        // relative to the canvas element
  		function ev_canvas(ev) {

    		if (ev.layerX || ev.layerX == 0) { // Firefox
      			ev._x = ev.layerX;
      			ev._y = ev.layerY;
    		} else if (ev.offsetX || ev.offsetX == 0) { // Opera
      			ev._x = ev.offsetX;
      			ev._y = ev.offsetY;
    		}

    		var func = tool[ev.type];
    		if (func) {
      			func(ev);
    		}
  		}

        // Convert the image to an input vector compatible with the artificial
        // neural network on the server
		function evalImage() {

			// Get image data
			var imageData = 
                context.getImageData(0, 0, canvas.width, canvas.height);
			var data = imageData.data;

			// Extract alpha values (there is no color)
			var matrix = new Array();
			for (var i = 3, j = 0; i < data.length; i += 4, j++) {
				matrix[j] = data[i];
			}
		
			// Shrink 308x308 image to 14*14
			var TARGET_SIZE = 14;
			var sampleSize = canvas.width / TARGET_SIZE;
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
            var iv = result.join();

            // Post to server for evaluation, then display result in UI
            $.get('eval?iv='+iv, function(data) {
                var words = {
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
                    };
                $("#classification").html(words[data.toString()]);
                $("#calloutBefore").hide();
                $("#calloutAfter").show();
            });

		}

        function clearImage() {
            context.clearRect(0, 0, canvas.width, canvas.height);
            $("#calloutAfter").hide();
            $("#calloutBefore").show();
        }

  		init();

	}, false); 
}

