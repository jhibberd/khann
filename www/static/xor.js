
function initWidget() {

    var onIVSubmit = function() {
        var sigA, sigB;
        sigA = $("#signalA").is(':checked') ? 1.0 : 0.0;
        sigB = $("#signalB").is(':checked') ? 1.0 : 0.0;
        khann.evalIV("xor", [sigA, sigB], onIVResponse);
    };

    var onIVResponse = function(data) {
        var result = data.ov_bin[0];
        var msg = result == 1 ? 
            "<b>Yes</b> they are exclusively OR." :
            "<b>No</b> they are not exclusively OR.";
        $("#result").html(msg);
    };

    var onResetWidget = function() {
        $("#signalA").prop('checked', false);
        $("#signalB").prop('checked', false);
    };

    khann.registerCallbacks(onResetWidget, null);
    $("#ivSubmit").click(onIVSubmit);

}

