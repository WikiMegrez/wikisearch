data = {
    images: []
}

document.addEventListener('DOMContentLoaded', function() {
    var elems = document.querySelectorAll('.materialboxed');
    var instances = M.Materialbox.init(elems);

    var app = new Vue({
        el: '#app',
        data: data
    });

});

function searchClicked() {
    input = document.getElementById('search_query');
    query = input.value;
    console.log(query);

    fetch("/search/?q=" + query)
        .then(resp => resp.json())
        .then(resp => {
            console.log(resp)
            data.images = resp
        })
}

