
function initWidget() {

    var onIVSubmit = function() {
        var numA, numB;
        numA = toBinArr(toBinStr($("#numA").val()));
        numB = toBinArr(toBinStr($("#numB").val()));
        khann.evalIV("binadd", numA.concat(numB), onIVResponse);
    };

    // Represent int as 5-digit binary string
    var toBinStr = function(x) {
        var b, missing;
        b  = parseInt(x).toString(2);
        missing = 5 - b.length;
        return Array(missing+1).join("0") + b;
    };

    // Convert a binary string to an array of binary ints
    var toBinArr = function(x) {
        var result = [];
        for (var i = 0; i < x.length; i++)
            result.push(parseInt(x[i]));
        return result;
    };

    var onIVResponse = function(data) {
        var sum = parseInt(data.ov_bin.join(''), 2);
        $("#sum").html(sum);
    };

    var onResetWidget = function() {
        $("#numA").val("0");
        $("#numB").val("0");
    };

    // Dynamically generate number lists
    var nums = [];
    for (var i = 0; i <= 30; i++)
        nums.push(i);
    $.each(nums, function (index, value) {
        $('#numA').append($('<option>', { 
            value: value,
            text : value 
        }));
        $('#numB').append($('<option>', { 
            value: value,
            text : value 
        }));
    });

    khann.registerCallbacks(onResetWidget, null);
    $("#ivSubmit").click(onIVSubmit);

}

