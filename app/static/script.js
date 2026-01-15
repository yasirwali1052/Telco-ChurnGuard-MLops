const API_BASE_URL = window.location.origin;

// Form elements
const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const resetBtn = document.getElementById('resetBtn');
const resultsCard = document.getElementById('resultsCard');
const loadingOverlay = document.getElementById('loadingOverlay');

// Results elements
const predictionBadge = document.getElementById('predictionBadge');
const predictionLabel = document.getElementById('predictionLabel');
const progressFill = document.getElementById('progressFill');
const churnProbability = document.getElementById('churnProbability');
const churnRisk = document.getElementById('churnRisk');
const retentionProb = document.getElementById('retentionProb');
const recommendationsList = document.getElementById('recommendationsList');

// Form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading
    showLoading();
    
    // Collect form data
    const formData = new FormData(form);
    const data = {};
    
    for (const [key, value] of formData.entries()) {
        data[key] = value;
    }
    
    // Convert SeniorCitizen to number
    data.SeniorCitizen = parseInt(data.SeniorCitizen);
    data.tenure = parseInt(data.tenure);
    data.MonthlyCharges = parseFloat(data.MonthlyCharges);
    
    // Handle empty TotalCharges
    if (!data.TotalCharges || data.TotalCharges.trim() === '') {
        data.TotalCharges = null;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error making prediction: ' + error.message);
    } finally {
        hideLoading();
    }
});

// Reset form
resetBtn.addEventListener('click', () => {
    form.reset();
    resultsCard.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Display results
function displayResults(result) {
    const prob = (result.churn_probability * 100).toFixed(1);
    const isChurn = result.prediction === 1;
    
    // Update prediction badge
    predictionBadge.className = `prediction-badge ${isChurn ? 'churn' : 'no-churn'}`;
    predictionBadge.innerHTML = `
        <i class="fas ${isChurn ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i>
        <span>${result.prediction_label}</span>
    `;
    
    // Update progress bar
    progressFill.style.width = `${prob}%`;
    churnProbability.textContent = `${prob}%`;
    
    // Update risk level
    let riskLevel = 'Low';
    let riskColor = 'success';
    if (prob >= 70) {
        riskLevel = 'High';
        riskColor = 'danger';
    } else if (prob >= 40) {
        riskLevel = 'Medium';
        riskColor = 'warning';
    }
    churnRisk.textContent = riskLevel;
    
    // Update retention probability
    const retention = (result.no_churn_probability * 100).toFixed(1);
    retentionProb.textContent = `${retention}%`;
    
    // Generate recommendations
    generateRecommendations(result, prob, riskLevel);
    
    // Show results card
    resultsCard.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// Generate recommendations
function generateRecommendations(result, prob, riskLevel) {
    const recommendations = [];
    
    if (prob >= 70) {
        recommendations.push('Immediate intervention required - High churn risk detected');
        recommendations.push('Consider offering a retention discount or special promotion');
        recommendations.push('Assign to customer success team for personalized outreach');
        recommendations.push('Review service quality and address any recent complaints');
    } else if (prob >= 40) {
        recommendations.push('Monitor customer engagement closely');
        recommendations.push('Consider proactive outreach to understand customer needs');
        recommendations.push('Review contract terms and offer upgrade options');
        recommendations.push('Send satisfaction survey to identify pain points');
    } else {
        recommendations.push('Customer shows low churn risk - maintain current service quality');
        recommendations.push('Consider upselling additional services or features');
        recommendations.push('Continue regular engagement and satisfaction checks');
    }
    
    // Add general recommendations based on probability
    if (prob >= 50) {
        recommendations.push('Review customer service history and address any issues');
        recommendations.push('Consider loyalty rewards or retention incentives');
    }
    
    // Display recommendations
    recommendationsList.innerHTML = recommendations
        .map(rec => `<li><i class="fas fa-check-circle"></i> ${rec}</li>`)
        .join('');
}

// Loading functions
function showLoading() {
    loadingOverlay.style.display = 'flex';
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
    predictBtn.disabled = false;
    predictBtn.innerHTML = '<i class="fas fa-brain"></i> Predict Churn';
}

// Check API health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            console.warn('API health check failed');
        }
    } catch (error) {
        console.error('Error checking API health:', error);
    }
});

