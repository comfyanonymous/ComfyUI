// research_web/js/panels/AssetPanel.js
/**Asset Library panel - browse and manage paper assets.*/

export function renderAssetPanel(container, api) {
    container.innerHTML = `
        <div class="asset-panel">
            <div class="asset-header">
                <h2>Asset Library</h2>
                <div class="asset-tabs">
                    <button class="tab-btn active" data-tab="papers">Papers</button>
                    <button class="tab-btn" data-tab="claims">Claims</button>
                    <button class="tab-btn" data-tab="sources">Sources</button>
                </div>
            </div>
            <div class="asset-content">
                <div class="asset-list" id="asset-list">
                    <p class="loading">Loading...</p>
                </div>
            </div>
        </div>
    `;

    // Tab switching
    container.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            container.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadAssets(btn.dataset.tab);
        });
    });

    // Initial load
    loadAssets('papers');

    function loadAssets(type) {
        const list = container.querySelector('#asset-list');
        if (type === 'papers') {
            list.innerHTML = '<p class="loading">Loading papers...</p>';
            api.listPapers().then(papers => {
                if (!papers || papers.length === 0) {
                    list.innerHTML = '<p class="empty-state">No papers in library. Search papers in ComfyUI canvas to add them.</p>';
                    return;
                }
                list.innerHTML = papers.map(p => `
                    <div class="asset-card paper-card" data-id="${p.id}" draggable="true">
                        <h4>${p.title || 'Untitled'}</h4>
                        <p class="asset-meta">${p.authors_text || 'Unknown authors'}</p>
                        <p class="asset-meta">${p.journal_or_source || ''} ${p.published_at || ''}</p>
                        <div class="asset-badges">
                            <span class="badge ${p.read_status}">${p.read_status || 'unread'}</span>
                            <span class="badge ${p.library_status}">${p.library_status || 'pending'}</span>
                        </div>
                        <div class="asset-actions">
                            <button class="btn-quick-read" data-id="${p.id}">Quick Read</button>
                            <button class="btn-drag" data-id="${p.id}" title="Drag to canvas">Drag to Canvas</button>
                        </div>
                    </div>
                `).join('');

                // Make cards draggable for canvas
                list.querySelectorAll('.asset-card').forEach(card => {
                    card.addEventListener('dragstart', (e) => {
                        e.dataTransfer.setData('application/json', JSON.stringify({
                            type: 'paper_asset',
                            id: card.dataset.id,
                            title: card.querySelector('h4').textContent
                        }));
                        e.dataTransfer.effectAllowed = 'copy';
                    });
                });
            }).catch(err => {
                list.innerHTML = `<p class="error">Failed to load: ${err.message}</p>`;
            });
        } else if (type === 'claims') {
            list.innerHTML = '<p class="empty-state">Claims panel coming in Phase 3.</p>';
        } else if (type === 'sources') {
            list.innerHTML = '<p class="loading">Loading sources...</p>';
            api.listSources().then(sources => {
                if (!sources || sources.length === 0) {
                    list.innerHTML = '<p class="empty-state">No sources configured. Add sources to start your feed.</p>';
                    return;
                }
                list.innerHTML = sources.map(s => `
                    <div class="asset-card source-card" data-id="${s.id}">
                        <h4>${s.name}</h4>
                        <p class="asset-meta">${s.category || ''} - ${s.intake_type}</p>
                        <div class="asset-badges">
                            <span class="badge ${s.enabled ? 'enabled' : 'disabled'}">${s.enabled ? 'enabled' : 'disabled'}</span>
                        </div>
                    </div>
                `).join('');
            }).catch(err => {
                list.innerHTML = `<p class="error">Failed to load: ${err.message}</p>`;
            });
        }
    }
}