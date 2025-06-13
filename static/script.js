
document.getElementById('inputFile').addEventListener('change', function () {
  const fileName = this.files[0]?.name || 'No file selected';
  document.getElementById('fileName').textContent = `Selected: ${fileName}`;
});

document.getElementById('processForm').addEventListener('submit', function (e) {
  e.preventDefault();

  const formData = new FormData();
  const fileInput = document.getElementById('inputFile');
  formData.append('file', fileInput.files[0]);

  document.getElementById('loadingSpinner').style.display = 'block';

  fetch('/process', {
    method: 'POST',
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      console.log("Backend response:",data);
      const predictionOutput = document.getElementById('predictionOutput');
      const row = document.createElement('div');
      row.className = 'row g-4';

      const container = document.getElementById('plotContainer');
      container.innerHTML = '';
      predictionOutput.innerHTML = '';

      if (data.error) {
        document.getElementById('errorToastBody').textContent = data.error;
        const toast = new bootstrap.Toast(document.getElementById('errorToast'));
        toast.show();
        return;
      }


            // Handle breakdowns
      if (data.breakdowns && data.breakdowns.length > 0) {
        data.breakdowns.forEach(b => {
          const col = document.createElement('div');
          col.className = 'col-md-6 fade-in';

          col.innerHTML = `
            <div class="card bg-dark text-light border-info shadow-sm h-100">
              <div class="card-body">
                <h5 class="card-title text-info d-flex align-items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#17a2b8" class="bi bi-exclamation-triangle-fill me-2" viewBox="0 0 16 16">
                    <path d="M8.982 1.566a1.13 1.13 0 0 0-1.964 0L.165 13.233c-.457.778.091 1.767.982 1.767h13.706c.89 0 1.438-.99.982-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 0 .002 2 1 1 0 0 0-.002-2z"/>
                  </svg>
                  Broken Detected
                </h5>
                <ul class="list-group list-group-flush mt-3">
                  <li class="list-group-item bg-transparent text-light d-flex justify-content-between">
                    <span><strong>Initial Detection</strong></span>
                    <span class="text-warning">${b.start || 'N/A'}</span>
                  </li>
                  <li class="list-group-item bg-transparent text-light d-flex justify-content-between">
                    <span><strong>Final Detection</strong></span>
                    <span class="text-warning">${b.end || 'N/A'}</span>
                  </li>
                  <li class="list-group-item bg-transparent text-light d-flex justify-content-between">
                    <span><strong>Duration</strong></span>
                    <span class="badge bg-info text-dark">${b.duration || 'N/A'}</span>
                  </li>
                </ul>
              </div>
            </div>
          `;
          row.appendChild(col);
        });
      }

      // Handle recoveries
      if (data.recoveries && data.recoveries.length > 0) {
        data.recoveries.forEach(r => {
          const col = document.createElement('div');
          col.className = 'col-md-6 fade-in';

          col.innerHTML = `
            <div class="card bg-dark text-light border-success shadow-sm h-100">
              <div class="card-body">
                <h5 class="card-title text-success d-flex align-items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#28a745" class="bi bi-check-circle-fill me-2" viewBox="0 0 16 16">
                    <path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM6.97 11.03a.75.75 0 0 0 1.08.022L11.03 8.5a.75.75 0 0 0-1.06-1.06L7.5 9.44 6.53 8.47a.75.75 0 0 0-1.06 1.06l1.5 1.5z"/>
                  </svg>
                  Recovering Detected
                </h5>
                <ul class="list-group list-group-flush mt-3">
                  <li class="list-group-item bg-transparent text-light d-flex justify-content-between">
                    <span><strong>Initial Detection</strong></span>
                    <span class="text-success">${r.start || 'N/A'}</span>
                  </li>
                  <li class="list-group-item bg-transparent text-light d-flex justify-content-between">
                    <span><strong>Final Detection</strong></span>
                    <span class="text-success">${r.end || 'N/A'}</span>
                  </li>
                  <li class="list-group-item bg-transparent text-light d-flex justify-content-between">
                    <span><strong>Duration</strong></span>
                    <span class="badge bg-success text-dark">${r.duration || 'N/A'}</span>
                  </li>
                </ul>
              </div>
            </div>
          `;
          row.appendChild(col);
        });
      }

      if (row.children.length > 0) {
        predictionOutput.appendChild(row);
      } else if (data.message) {
        predictionOutput.innerHTML = `
          <div class="alert alert-success fade-in" role="alert">
            ${data.message}
          </div>
        `;
      }

      // Render sensor and FFT plots
      if (data.plot_urls && data.fft_urls) {
        for (let i = 0; i < data.plot_urls.length; i++) {
          const sensorNumber = i + 1;

          const row = document.createElement('div');
          row.className = 'plot-pair mb-4 fade-in';

          const sensorPlot = document.createElement('div');
          sensorPlot.className = 'plot-box';
          sensorPlot.innerHTML = `
            <h6>Sensor ${sensorNumber} – Time Series</h6>
            <img src="${data.plot_urls[i]}" alt="Sensor Plot" class="img-fluid rounded border shadow">
          `;

          const fftPlot = document.createElement('div');
          fftPlot.className = 'plot-box';
          fftPlot.innerHTML = `
            <h6>Sensor ${sensorNumber} – FFT Spectrum</h6>
            <img src="${data.fft_urls[i]}" alt="FFT Plot" class="img-fluid rounded border shadow">
          `;

          row.appendChild(sensorPlot);
          row.appendChild(fftPlot);
          container.appendChild(row);
        }
      }
    })
    .catch(err => {
      document.getElementById('errorToastBody').textContent = err.message;
      const toast = new bootstrap.Toast(document.getElementById('errorToast'));
      toast.show();
    })
    .finally(() => {
      document.getElementById('loadingSpinner').style.display = 'none';
    });
});
