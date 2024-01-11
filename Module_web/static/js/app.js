const form = document.querySelector('form')
const btn = document.querySelector('form button')
const Country = document.querySelector('form #Country')
const year = document.querySelector('form #year')
const Income = document.querySelector('form #Income')
const schooling = document.querySelector('form #schooling')
const result = document.querySelector('.result')
const url = "http://127.0.0.1:5000/"
let issendding = false;
form.addEventListener('submit', function(e){
    e.preventDefault()
    if (issendding) return;
    issendding = !issendding

    // isloading
    btn.innerHTML = "Chargement..."

    // get form data
    // perform request
    fetch(
        url,
        {
            method: "POST",
            headers: {
                "Content-type": "application/json"
            },
            body: JSON.stringify({
                "Country": Country.value,
                "year": year.value,
                "Income": Income.value,
                "schooling": schooling.value
            })
        }
    ).then(
        async function (data) {
            btn.innerHTML = "prediction"
            issendding = !issendding
            if (data.status == 200) {
                let res = await data.json()
                
                result.innerHTML = `L'esperance de vie au ${Country.value} en ${year.value} est de ${Math.round(+res.response)}ans.`
            }else{
                result.innerHTML = "une erreur est survenue !"
            }
        }
    ).catch(
        function (er) {
            btn.innerHTML = "prediction"
            issendding = !issendding
            console.log(er)
        }
    )

    
})
