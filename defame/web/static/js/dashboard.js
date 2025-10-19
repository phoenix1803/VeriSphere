// VeriSphere Dashboard JavaScript

class VeriSphereDashboard {
    constructor() {
        this.apiBase = '/api/v1';
        this.currentUser = null;
        this.activeClaims = new Map();
        this.refreshInterval = null;

        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.setupTabs();
        await this.loadSystemStatus();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        // Claim submission form
        const claimForm = document.getElementById('claim-form');
        if (claimForm) {
            claimForm.addEventListener('submit', (e) => this.handleClaimSubmission(e));
        }

        // Refresh buttons
        document.querySelectorAll('.refresh-btn').forEach(btn => {
            btn.addEventListener('click', () => this.refreshData());
        });

        // Auto-refresh toggle
        const autoRefreshToggle = document.getElementById('auto-refresh');
        if (autoRefreshToggle) {
            autoRefreshToggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.startAutoRefresh();
                } else {
                    this.stopAutoRefresh();
                }
            });
        }
    }

    setupTabs() {
        const tabButtons = document.querySelectorAll('.nav-tabs li');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.dataset.tab;

                // Update active tab button
                tabButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');

                // Update active tab content
                tabContents.forEach(content => {
                    content.classList.remove('active');
                    if (content.id === targetTab) {
                        content.classList.add('active');
                    }
                });

                // Load tab-specific data
                this.loadTabData(targetTab);
            });
        });
    }

    async loadTabData(tabId) {
        switch (tabId) {
            case 'dashboard':
                await this.loadSystemStatus();
                await this.loadRecentClaims();
                break;
            case 'submit-claim':
                // No additional data needed
                break;
            case 'claims':
                await this.loadAllClaims();
                break;
            case 'system':
                await this.loadSystemMetrics();
                break;
        }
    }

    async handleClaimSubmission(event) {
        event.preventDefault();

        const form = event.target;
        const submitBtn = form.querySelector('button[type="submit"]');
        const originalText = submitBtn.textContent;

        // Show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner"></span> Submitting...';

        try {
            const formData = new FormData(form);
            const claimData = {
                content: formData.get('content'),
                claim_type: formData.get('claim_type'),
                priority: formData.get('priority'),
                source: formData.get('source') || null,
                metadata: {}
            };

            const response = await this.apiCall('POST', '/claims', claimData);

            if (response.claim_id) {
                this.showAlert('success', `Claim submitted successfully! ID: ${response.claim_id}`);
                form.reset();

                // Switch to claims tab to show progress
                this.switchTab('claims');
                await this.loadAllClaims();
            }
        } catch (error) {
            this.showAlert('error', `Failed to submit claim: ${error.message}`);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = originalText;
        }
    }

    async loadSystemStatus() {
        try {
            const status = await this.apiCall('GET', '/status');
            this.updateSystemStats(status);
        } catch (error) {
            console.error('Failed to load system status:', error);
            this.showAlert('error', 'Failed to load system status');
        }
    }

    async loadRecentClaims() {
        try {
            const claims = await this.apiCall('GET', '/claims/active');
            this.updateRecentClaims(claims);
        } catch (error) {
            console.error('Failed to load recent claims:', error);
        }
    }

    async loadAllClaims() {
        try {
            const claims = await this.apiCall('GET', '/claims/active');
            this.updateClaimsTable(claims);
        } catch (error) {
            console.error('Failed to load claims:', error);
            this.showAlert('error', 'Failed to load claims');
        }
    }

    async loadSystemMetrics() {
        try {
            const metrics = await this.apiCall('GET', '/metrics');
            this.updateSystemMetrics(metrics);
        } catch (error) {
            console.error('Failed to load system metrics:', error);
        }
    }

    updateSystemStats(status) {
        // Update stat cards
        this.updateStatCard('total-agents', status.agents?.total_agents || 0);
        this.updateStatCard('healthy-agents', status.agents?.healthy_agents || 0);
        this.updateStatCard('active-claims', status.active_pipelines || 0);
        this.updateStatCard('total-processed', status.total_claims_processed || 0);

        // Update agent status
        const agentsList = document.getElementById('agents-list');
        if (agentsList && status.agents?.agents) {
            agentsList.innerHTML = '';

            Object.entries(status.agents.agents).forEach(([agentId, agent]) => {
                const agentElement = document.createElement('div');
                agentElement.className = 'agent-status';
                agentElement.innerHTML = `
                    <div class="agent-info">
                        <strong>${agent.agent_type.replace('_', ' ').toUpperCase()}</strong>
                        <span class="badge ${agent.is_healthy ? 'badge-success' : 'badge-error'}">
                            ${agent.is_healthy ? 'Healthy' : 'Unhealthy'}
                        </span>
                    </div>
                    <div class="agent-metrics">
                        <small>Processed: ${agent.metrics?.processing_count || 0}</small>
                        <small>Errors: ${agent.metrics?.error_count || 0}</small>
                    </div>
                `;
                agentsList.appendChild(agentElement);
            });
        }
    }

    updateStatCard(cardId, value) {
        const card = document.getElementById(cardId);
        if (card) {
            const valueElement = card.querySelector('.stat-value');
            if (valueElement) {
                valueElement.textContent = value.toLocaleString();
            }
        }
    }

    updateRecentClaims(claimsData) {
        const container = document.getElementById('recent-claims');
        if (!container) return;

        if (!claimsData.claims || Object.keys(claimsData.claims).length === 0) {
            container.innerHTML = '<p class="text-secondary">No active claims</p>';
            return;
        }

        container.innerHTML = '';

        Object.entries(claimsData.claims).forEach(([claimId, claim]) => {
            const claimElement = document.createElement('div');
            claimElement.className = 'claim-item';
            claimElement.innerHTML = `
                <div class="claim-header">
                    <strong>${claimId.substring(0, 8)}...</strong>
                    <span class="badge badge-primary">${claim.current_stage}</span>
                </div>
                <div class="claim-progress">
                    <div class="progress">
                        <div class="progress-bar" style="width: ${(claim.progress * 100).toFixed(1)}%"></div>
                    </div>
                    <small>${(claim.progress * 100).toFixed(1)}% complete</small>
                </div>
            `;
            container.appendChild(claimElement);
        });
    }

    updateClaimsTable(claimsData) {
        const tbody = document.querySelector('#claims-table tbody');
        if (!tbody) return;

        tbody.innerHTML = '';

        if (!claimsData.claims || Object.keys(claimsData.claims).length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-secondary">No active claims</td></tr>';
            return;
        }

        Object.entries(claimsData.claims).forEach(([claimId, claim]) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${claimId.substring(0, 12)}...</td>
                <td><span class="badge badge-primary">${claim.current_stage}</span></td>
                <td>
                    <div class="progress">
                        <div class="progress-bar" style="width: ${(claim.progress * 100).toFixed(1)}%"></div>
                    </div>
                </td>
                <td>${claim.processing_time.toFixed(1)}s</td>
                <td>
                    <button class="btn btn-sm btn-secondary" onclick="dashboard.viewClaimDetails('${claimId}')">
                        View
                    </button>
                    <button class="btn btn-sm btn-error" onclick="dashboard.cancelClaim('${claimId}')">
                        Cancel
                    </button>
                </td>
            `;
            tbody.appendChild(row);
        });
    }

    async viewClaimDetails(claimId) {
        try {
            const status = await this.apiCall('GET', `/claims/${claimId}/status`);

            // Create modal or detailed view
            const modal = this.createModal('Claim Details', `
                <div class="claim-details">
                    <h4>Claim ID: ${claimId}</h4>
                    <p><strong>Stage:</strong> ${status.current_stage}</p>
                    <p><strong>Progress:</strong> ${(status.progress * 100).toFixed(1)}%</p>
                    <p><strong>Processing Time:</strong> ${status.processing_time.toFixed(1)}s</p>
                    <p><strong>Agents:</strong> ${status.agent_results_count}</p>
                    ${status.errors_count > 0 ? `<p><strong>Errors:</strong> ${status.errors_count}</p>` : ''}
                </div>
            `);

            document.body.appendChild(modal);
        } catch (error) {
            this.showAlert('error', `Failed to load claim details: ${error.message}`);
        }
    }

    async cancelClaim(claimId) {
        if (!confirm('Are you sure you want to cancel this claim?')) {
            return;
        }

        try {
            await this.apiCall('DELETE', `/claims/${claimId}`);
            this.showAlert('success', 'Claim cancelled successfully');
            await this.loadAllClaims();
        } catch (error) {
            this.showAlert('error', `Failed to cancel claim: ${error.message}`);
        }
    }

    switchTab(tabId) {
        const tabButton = document.querySelector(`[data-tab="${tabId}"]`);
        if (tabButton) {
            tabButton.click();
        }
    }

    createModal(title, content) {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${title}</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
            </div>
        `;

        // Close modal functionality
        modal.querySelector('.modal-close').addEventListener('click', () => {
            document.body.removeChild(modal);
        });

        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                document.body.removeChild(modal);
            }
        });

        return modal;
    }

    showAlert(type, message) {
        const alertContainer = document.getElementById('alerts');
        if (!alertContainer) return;

        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.textContent = message;

        alertContainer.appendChild(alert);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.parentNode.removeChild(alert);
            }
        }, 5000);
    }

    async apiCall(method, endpoint, data = null) {
        const url = `${this.apiBase}${endpoint}`;
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(url, options);

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return await response.json();
    }

    startAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }

        this.refreshInterval = setInterval(() => {
            this.refreshData();
        }, 10000); // Refresh every 10 seconds
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    async refreshData() {
        const activeTab = document.querySelector('.tab-content.active');
        if (activeTab) {
            await this.loadTabData(activeTab.id);
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new VeriSphereDashboard();
});

// Add modal styles
const modalStyles = `
.modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}

.modal-content {
    background: white;
    border-radius: 8px;
    max-width: 600px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
}

.modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem;
    border-bottom: 1px solid var(--border-color);
}

.modal-close {
    background: none;
    border: none;
    font-size: 1.5rem;
    cursor: pointer;
    color: var(--text-secondary);
}

.modal-body {
    padding: 1.5rem;
}

.claim-details h4 {
    margin-bottom: 1rem;
    color: var(--primary-color);
}

.claim-details p {
    margin-bottom: 0.5rem;
}

.agent-status {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    margin-bottom: 0.5rem;
}

.agent-info {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.agent-metrics {
    display: flex;
    gap: 1rem;
    font-size: 0.75rem;
    color: var(--text-secondary);
}

.claim-item {
    padding: 1rem;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    margin-bottom: 0.75rem;
}

.claim-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.claim-progress {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.claim-progress .progress {
    flex: 1;
}
`;

// Inject modal styles
const styleSheet = document.createElement('style');
styleSheet.textContent = modalStyles;
document.head.appendChild(styleSheet);