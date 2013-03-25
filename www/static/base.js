
var khann = {

    registerCallbacks: function(onResetWidget, onOVSubmit) {

        $("#playAgain").click(function() {
            onResetWidget();
            $("#boxClassification").hide();
            $("#boxInstruction").show();
        });

        $("#train").click(function() {
            $("#boxClassification").hide();
            $("#boxTrain").show();
        });

        $("#ovSubmit").click(onOVSubmit);
    },

    evalIV: function(nid, iv, onIVResponse) {
        $.get("//local.api.khann.org/"+nid+"/?iv="+iv.join(), function(data) {

            // Page to handle network response
            onIVResponse(data);
            
            // Hide instruction box and show classification box
            $("#boxInstruction").hide();
            $("#boxClassification").show();
        });
    },

    submitIV: function(nid, iv, ov) {
        var data = "iv=" + iv.join() + "&ov=" + ov.join();
        $.post("//local.api.khann.org/"+nid, data, function(data) {

            // Hide train box and show train thanks box
            $("#boxTrain").hide();
            $("#boxTrainThanks").show();
        });
    }

};

$(document).ready(function() {
    initWidget();
});

