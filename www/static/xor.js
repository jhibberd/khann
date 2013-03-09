
$(document).ready(function() {

    var onIVSubmit = function() {
        var sigA, sigB;
        sigA = $("#signalA").is(':checked') ? 1.0 : 0.0;
        sigB = $("#signalB").is(':checked') ? 1.0 : 0.0;
        khann.evalIV("xor", [sigA, sigB], onIVResponse);
    };

    var onIVResponse = function(data) {
        var result = data.ov_bin[0];
        alert(result == 1 ? "ON" : "OFF");
    };

    $("#ivSubmit").click(onIVSubmit);

});

