        document.addEventListener("DOMContentLoaded", function () {
            document.querySelectorAll(".predict-btn").forEach(btn => {
                btn.addEventListener("click", function () {
                    const customerID = this.dataset.customerid;
                    const rowID = this.dataset.rowid;
                    const resultCell = document.getElementById("result-" + rowID);
                    resultCell.textContent = "Loading...";

                    fetch('/predict?customerID=' + customerID, { method: 'POST' })
                        .then(res => res.text())
                        .then(result => {
                            resultCell.textContent = result;
                        })
                        .catch(err => {
                            resultCell.textContent = "Error";
                            console.error(err);
                        });
                });
            });
        });


    function showLoadingMessage() {
        const message = document.getElementById("loading-message");
        if (message) {
        message.style.display = "block";
        }
        // Automatically hide spinner after 5 seconds
        setTimeout(() => {
            message.style.display = "none";
        }, 1000);
    }