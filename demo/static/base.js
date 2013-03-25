
var khann = {

    registerCallbacks: function(onResetWidget, onOVSubmit) {

        $("#playAgain").click(function() {
            onResetWidget();
            $("#youDrew").hide();
            $("#drawSomething").show();
        });

    },

    evalIV: function(nid, iv, onIVResponse) {
        $.get("/eval/"+nid+"/?iv="+iv.join(), function(data) {

            // Page to handle network response
            onIVResponse(data);
            
            // Hide instruction box and show classification box
            $("#drawSomething").hide();
            $("#youDrew").show();
        });
    },

};

$(document).ready(function() {
    initWidget();
});

