
// Keep everything in anonymous function, called on window load.
if (window.addEventListener) {

	window.addEventListener('load', function() {
  		var canvas, context, tool;

  		function init () {
    		// Find the canvas element.
    		canvas = document.getElementById('imageView');
    		if (!canvas) {
      			alert('Error: I cannot find the canvas element!');
      			return;
    		}

    		if (!canvas.getContext) {
      			alert('Error: no canvas.getContext!');
      			return;
    		}

    		// Get the 2D canvas context.
    		context = canvas.getContext('2d');
    		if (!context) {
      			alert('Error: failed to getContext!');
      			return;
    		}

    		// Pencil tool instance.
    		tool = new tool_pencil();

    		// Attach the mousedown, mousemove and mouseup event listeners.
    		canvas.addEventListener('mousedown', ev_canvas, false);
    		canvas.addEventListener('mousemove', ev_canvas, false);
    		canvas.addEventListener('mouseup',   ev_canvas, false);
			canvas.addEventListener('mouseout',  on_mouseout, false);
  		}

  		// This painting tool works like a drawing pencil which tracks the mouse 
  		// movements.
  		function tool_pencil () {
    		var tool = this;
    		this.started = false;

    		context.lineWidth = 20;
    		context.lineCap = "round";

    		// This is called when you start holding down the mouse button.
    		// This starts the pencil drawing.
    		this.mousedown = function (ev) {
        		context.beginPath();
        		context.moveTo(ev._x, ev._y);
        		tool.started = true;
    		};

    		// This function is called every time you move the mouse. Obviously, it only 
    		// draws if the tool.started state is set to true (when you are holding down 
    		// the mouse button).
    		this.mousemove = function (ev) {
      			if (tool.started) {
        			context.lineTo(ev._x, ev._y);
        			context.stroke();
      			}
    		};

    		// This is called when you release the mouse button.
    		this.mouseup = function (ev) {
      			if (tool.started) {
        			tool.mousemove(ev);
        			tool.started = false;
      			}
    		};
  		}

  		// The general-purpose event handler. This function just determines the mouse 
  		// position relative to the canvas element.
  		function ev_canvas(ev) {
    		if (ev.layerX || ev.layerX == 0) { // Firefox
      			ev._x = ev.layerX;
      			ev._y = ev.layerY;
    		} else if (ev.offsetX || ev.offsetX == 0) { // Opera
      			ev._x = ev.offsetX;
      			ev._y = ev.offsetY;
    		}

    		// Call the event handler of the tool.
    		var func = tool[ev.type];
    		if (func) {
      			func(ev);
    		}
  		}

		function on_mouseout(ev) {

			// Get image data
			var imageData = context.getImageData(0, 0, canvas.width, canvas.height);
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

            $.get('eval?iv='+iv, function(data) {
                alert(data);
            });


			/* Print output
			for (var i = 0; i < result.length; i += 14) {
				var line = new Array();
				for (var j = 0; j < 14; j++) {
					var x = result[i+j];
					line[j] = x > 0 ? "x" : " ";
				}
				console.log(line);
			}
            */

		}

		function get_xy(a, w, x, y) {
			return a[(y * w) + x]
		}

  		init();

	}, 
	false); 
}

