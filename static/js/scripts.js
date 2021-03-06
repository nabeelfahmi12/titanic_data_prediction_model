/*!
    * Start Bootstrap - SB Admin v6.0.1 (https://startbootstrap.com/templates/sb-admin)
    * Copyright 2013-2020 Start Bootstrap
    * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-sb-admin/blob/master/LICENSE)
    */
function verifyFile(input) {
    var file = input.files[0];
    console.log(file)
    let y = (document.getElementById("source_data"));
    /* Extract the selected source type from dropdown*/
    let e = y.options[y.selectedIndex].value;

    /* Extract the extension from file choosen*/
    let x = (file.name.split("."));
    let extension = x[x.length - 1]
    if (e.toLowerCase() != extension.toLowerCase()) {
        document.getElementById("Error_1").innerHTML = "File type do not match";
        document.getElementById("Error_1").style.color = "Red";
    } else {
        document.getElementById("Error_1").innerHTML = "Successfully validated"
        document.getElementById("Error_1").style.color = "Green";
        //loadFile();
    }


    console.log(e, x[x.length - 1])
}

function loadFile() {
    //if(document.getElementById("Error_1").innerHTML== "")
    //{console.
    if (document.getElementById("Error_1").innerHTML == "") {
        document.getElementById("Error_1").innerHTML = "No file selected";
        document.getElementById("Error_1").style.color = "Red";
    } else {
        var x = "test.html"
        document.getElementById('showData1').innerHTML = "<h3><b> Loaded Data: </b></h3>" + '<object type = "text/html" id= "df1" data = "test.html"></object>';


        // Converting JSON-encoded string to JS object
        var json = '{"rows":"20", "columns": 10, "size": "20 MB"}';

        obj = JSON.parse(json);
        document.getElementById("showData2").innerHTML = "<p id='dfdetail'><h3> <b><u>Data Profiler</u></b></h3>" +
            "Number of rows: " + obj.rows + "<br>" +
            "Number of columns: " + obj.columns + "<br>" +
            "RAM Size: " + obj.size + "</p>";

        console.log(obj.rows, obj.columns, obj.size);

        var column_data = ["longitude", "latitude", "house_price"];

        for (var i of column_data) {
            var option = document.createElement('option');
            console.log(i);
            option.text = i;
            document.querySelector('#tgv').add(option, null);
        }
    }


}


function tableFromJson() {
    // the json data. (you can change the values for output.)
    var myJSON = [
        {
            'Model Name': 'Regression',
            'Accuracy': '78.54%', 'Precision': '85%'
        }
    ]

    var col = [];
    for (var i = 0; i < myJSON.length; i++) {
        for (var key in myJSON[i]) {
            if (col.indexOf(key) === -1) {
                col.push(key);
            }
        }
    }

    // Create a table.
    var table = document.createElement("table");

    // Create table header row using the extracted headers above.
    var tr = table.insertRow(-1);                   // table row.

    for (var i = 0; i < col.length; i++) {
        var th = document.createElement("th");      // table header.
        th.innerHTML = col[i];
        tr.appendChild(th);
    }

    // add json data to the table as rows.
    for (var i = 0; i < myJSON.length; i++) {

        tr = table.insertRow(-1);

        for (var j = 0; j < col.length; j++) {
            var tabCell = tr.insertCell(-1);
            tabCell.innerHTML = myJSON[i][col[j]];
        }
    }

    // Now, add the newly created table with json data, to a container.
    var divShowData = document.getElementById('showData');
    divShowData.innerHTML = "";
    divShowData.appendChild(table);

}

function getChange() {

    if (document.getElementById("Error_1").innerHTML != "") {
        var sel = document.getElementById('scripts');
        var el = document.getElementById('display');
        var maincontainer = $('#maincontainer');
        maincontainer.addClass('container');
        var text = document.createElement("P");

        var t = document.createTextNode("Graph For Visualization");
        text.appendChild(t);
        maincontainer.append(text);

        var file = '/img/';
        for (var i = 0; i <= 1; i++) {
            var im = document.createElement("img");
            im.src = 'img/test' + i + ".png";
            im.style.height = "350px";
            im.style.width = "500px";
            maincontainer.append(im);

        }

        el.value = sel.options[sel.selectedIndex].text;

        document.getElementById("showTxt").disabled = true;
        tableFromJson();
    } else {
        document.getElementById("Error_1").innerHTML = "No file selected";
        document.getElementById("Error_1").style.color = "Red";
    }

}

(function ($) {
    "use strict";

    // Add active state to sidbar nav links
    var path = window.location.href; // because the 'href' property of the DOM element is the absolute path
    $("#layoutSidenav_nav .sb-sidenav a.nav-link").each(function () {
        if (this.href === path) {
            $(this).addClass("active");
        }
    });

    // Toggle the side navigation
    $("#sidebarToggle").on("click", function (e) {
        e.preventDefault();
        $("body").toggleClass("sb-sidenav-toggled");
    });
})(jQuery);

var input = document.getElementById( 'file-upload' );
var infoArea = document.getElementById( 'file-upload-filename' );

input.addEventListener( 'change', showFileName );

function showFileName( event ) {

  // the change event gives us the input it occurred in
  var input = event.srcElement;

  // the input has an array of files in the `files` property, each one has a name that you can use. We're just using the name here.
  var fileName = input.files[0].name;

  // use fileName however fits your app best, i.e. add it into a div
  infoArea.textContent = 'File name: ' + fileName;
}