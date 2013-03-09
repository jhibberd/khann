
var khann = {

    evalIV: function(nid, iv, callback) {
        $.get("//local.api.khann.org/"+nid+"/?iv="+iv.join(), callback);
    }

};

