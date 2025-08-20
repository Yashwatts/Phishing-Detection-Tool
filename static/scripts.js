document.getElementById('phishing-form').onsubmit = async function (e) {
    e.preventDefault();  // Prevent form submission from reloading the page

    // Get the email content from the form
    const emailContent = document.querySelector('textarea[name="email_content"]').value;

    // Send a POST request to the Flask backend with the email content
    const response = await fetch('/predict', {
        method: 'POST',
        body: new URLSearchParams({ email_content: emailContent }),
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    });

    // Parse the JSON response from the backend
    const result = await response.json();

    // Display the result with percentage
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `Prediction: ${result.label} <br> Percentage: ${result.confidence}%`;
};
