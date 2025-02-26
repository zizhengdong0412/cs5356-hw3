const dogImage = document.getElementById("dog-image");
const fetchDogButton = document.getElementById("fetch-dog");
function fetchRandomDog() {
fetch("https://dog.ceo/api/breeds/image/random")
.then(response => response.json())
.then(data => {
dogImage.src = data.message;
})
}
fetchRandomDog();

// Click it. New Dog
fetchDogButton.addEventListener("click", fetchRandomDog);