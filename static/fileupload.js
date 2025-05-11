document.querySelectorAll(".drop-zone__input").forEach(inputElement => {
    const dropZoneElement = inputElement.closest(".drop_zone");
    dropZoneElement.addEventListener("click", ()=>{
        inputElement.click();
    })
})