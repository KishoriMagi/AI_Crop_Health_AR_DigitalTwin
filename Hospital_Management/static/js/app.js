document.addEventListener('DOMContentLoaded', function() {
    // Save Patient
    const savePatientBtn = document.getElementById('savePatientBtn');
    if (savePatientBtn) {
        savePatientBtn.addEventListener('click', function() {
            const form = document.getElementById('addPatientForm');
            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());

            fetch('/api/patients', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                if(result.success) {
                    location.reload();
                } else {
                    alert('Error adding patient: ' + result.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred.');
            });
        });
    }

    // Save Test Request
    const saveTestBtn = document.getElementById('saveTestBtn');
    if (saveTestBtn) {
        saveTestBtn.addEventListener('click', function() {
            const form = document.getElementById('addTestForm');
            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());
            
            // disable button
            saveTestBtn.disabled = true;
            saveTestBtn.textContent = "Sending...";

            fetch('/api/test-requests', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                if(result.success) {
                    location.reload();
                } else {
                    alert('Error creating test request: ' + result.error);
                    saveTestBtn.disabled = false;
                    saveTestBtn.textContent = "Send Request";
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred.');
                saveTestBtn.disabled = false;
                saveTestBtn.textContent = "Send Request";
            });
        });
    }

    // Sync Results
    const syncResultsBtn = document.getElementById('syncResultsBtn');
    if (syncResultsBtn) {
        syncResultsBtn.addEventListener('click', function() {
            syncResultsBtn.disabled = true;
            syncResultsBtn.textContent = 'Syncing...';
            
            fetch('/api/dropbox/sync-results', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(result => {
                if(result.success) {
                    alert('Synced ' + result.synced_count + ' results.');
                    location.reload();
                } else {
                    alert('Error syncing results.');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred.');
            })
            .finally(() => {
                syncResultsBtn.disabled = false;
                syncResultsBtn.textContent = 'Sync Results from Lab';
            });
        });
    }

    // View Result Modal
    const viewResultBtns = document.querySelectorAll('.view-result-btn');
    const resultModalBody = document.getElementById('resultModalBody');
    const resultModal = new bootstrap.Modal(document.getElementById('resultModal'));
    
    viewResultBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const testId = this.getAttribute('data-id');
            resultModalBody.innerHTML = 'Loading...';
            resultModal.show();
            
            fetch('/api/results/' + testId)
            .then(response => response.json())
            .then(result => {
                if (result.error) {
                    resultModalBody.innerHTML = '<div class="alert alert-danger">' + result.error + '</div>';
                } else {
                    let html = '<table class="table table-bordered"><tbody>';
                    for (const [key, value] of Object.entries(result)) {
                        html += `<tr><th>${key}</th><td>${value}</td></tr>`;
                    }
                    html += '</tbody></table>';
                    resultModalBody.innerHTML = html;
                }
            })
            .catch(error => {
                resultModalBody.innerHTML = '<div class="alert alert-danger">Error fetching result</div>';
            });
        });
    });
});
