/* Agent Graph Canvas — drag-and-drop visual editor for agent workflows.
 *
 * Implements a minimal node editor with drag from palette, connection,
 * selection, property editing, and graph persistence.
 */

let nodes = {};
let edges = [];
let selectedNodeId = null;
let nextNodeId = 1;
let panX = 0, panY = 0;
let isDragging = false, dragNodeId = null, dragOffsetX = 0, dragOffsetY = 0;
let isPanning = false, panStartX = 0, panStartY = 0;
let tempEdge = null; // {source, mouseX, mouseY} while drawing a connection

const NODE_WIDTH = 160;
const NODE_HEIGHT = 60;
const NODE_COLORS = {
    start: '#22c55e', llm: '#3b82f6', tool: '#f59e0b',
    condition: '#8b5cf6', loop: '#ec4899', end: '#ef4444'
};
const svg = document.getElementById('canvas-svg');
const container = document.getElementById('canvas-container');

// ── Initialization ──

function init() {
    render();
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Delete' || e.key === 'Backspace') {
            if (selectedNodeId && !e.target.closest('input,select,textarea')) {
                deleteNode(selectedNodeId);
            }
        }
        if (e.key === 'Escape') {
            selectedNodeId = null;
            tempEdge = null;
            updateProperties();
            render();
        }
        if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            // Could add undo here
        }
    });
    // Canvas mouse events
    container.addEventListener('mousedown', onCanvasMouseDown);
    container.addEventListener('mousemove', onCanvasMouseMove);
    container.addEventListener('mouseup', onCanvasMouseUp);
    container.addEventListener('dblclick', onCanvasDblClick);
    // Load saved graphs on init
    refreshGraphList();
}

// ── Node Management ──

function addNode(type, x, y) {
    const id = 'node_' + (nextNodeId++);
    const defaults = {
        start: { label: 'Start', system_prompt: '' },
        llm: { label: 'LLM Think', model: 'qwen3.5-9b', temperature: 0.7, max_tokens: 4096, system_prompt: '' },
        tool: { label: 'Tool Call', tool_name: 'file_read', tool_params: {} },
        condition: { label: 'Condition', condition_expr: 'true' },
        loop: { label: 'Loop', max_iterations: 10 },
        end: { label: 'End' }
    };
    nodes[id] = { id, type, x, y, ...defaults[type] || {} };
    hideNoGraphMsg();
    render();
    selectNode(id);
    return id;
}

function deleteNode(id) {
    if (!nodes[id]) return;
    edges = edges.filter(e => e.source !== id && e.target !== id);
    delete nodes[id];
    if (selectedNodeId === id) { selectedNodeId = null; updateProperties(); }
    render();
    if (Object.keys(nodes).length === 0) showNoGraphMsg();
}

function selectNode(id) {
    selectedNodeId = id;
    updateProperties();
    render();
}

// ── Edge Management ──

function addEdge(source, target) {
    if (source === target) return;
    if (edges.some(e => e.source === source && e.target === target)) return;
    edges.push({ source, target, label: '' });
    render();
}

function deleteEdge(source, target) {
    edges = edges.filter(e => !(e.source === source && e.target === target));
    render();
}

// ── Rendering ──

function render() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    let html = '<defs>'
        + '<marker id="ah" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">'
        + '<polygon points="0 0,10 3.5,0 7" fill="' + (isDark ? '#475569' : '#94a3b8') + '"/></marker>'
        + '</defs>';

    // Draw edges
    edges.forEach(e => {
        const src = nodes[e.source], tgt = nodes[e.target];
        if (!src || !tgt) return;
        const sx = src.x + NODE_WIDTH / 2 + panX, sy = src.y + NODE_HEIGHT + panY;
        const tx = tgt.x + NODE_WIDTH / 2 + panX, ty = tgt.y + panY;
        const cy = (sy + ty) / 2;
        html += `<path class="canvas-edge" d="M${sx},${sy} C${sx},${cy} ${tx},${cy} ${tx},${ty}" marker-end="url(#ah)"/>`;
        if (e.label) {
            html += `<text class="edge-label" x="${(sx + tx) / 2}" y="${cy - 4}" text-anchor="middle">${escapeHtml(e.label)}</text>`;
        }
    });

    // Draw temp edge while dragging
    if (tempEdge) {
        const src = nodes[tempEdge.source];
        if (src) {
            const sx = src.x + NODE_WIDTH / 2 + panX, sy = src.y + NODE_HEIGHT + panY;
            html += `<line x1="${sx}" y1="${sy}" x2="${tempEdge.mx}" y2="${tempEdge.my}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="5,5"/>`;
        }
    }

    // Draw nodes
    Object.values(nodes).forEach(n => {
        const cx = n.x + panX, cy = n.y + panY;
        const color = NODE_COLORS[n.type] || '#64748b';
        const selected = n.id === selectedNodeId;
        const stroke = selected ? '#3b82f6' : 'transparent';
        const strokeWidth = selected ? 2 : 0;
        html += `<g class="canvas-node" data-id="${n.id}" transform="translate(${cx},${cy})">`;
        // Connection point (bottom)
        if (n.type !== 'end') {
            html += `<circle cx="${NODE_WIDTH / 2}" cy="${NODE_HEIGHT}" r="6" fill="${color}" stroke="#fff" stroke-width="2" class="connector-out" data-id="${n.id}"/>`;
        }
        // Connection point (top)
        if (n.type !== 'start') {
            html += `<circle cx="${NODE_WIDTH / 2}" cy="0" r="6" fill="${color}" stroke="#fff" stroke-width="2" class="connector-in" data-id="${n.id}"/>`;
        }
        html += `<rect x="0" y="0" width="${NODE_WIDTH}" height="${NODE_HEIGHT}" fill="${color}" opacity="0.9" stroke="${stroke}" stroke-width="${strokeWidth}"/>`;
        html += `<text x="${NODE_WIDTH / 2}" y="28" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">${escapeHtml(n.label || n.type)}</text>`;
        html += `<text x="${NODE_WIDTH / 2}" y="46" text-anchor="middle" fill="rgba(255,255,255,0.7)" font-size="10">${n.type.toUpperCase()}</text>`;
        html += '</g>';
    });

    svg.innerHTML = html;
    bindNodeEvents();
}

function bindNodeEvents() {
    // Node click
    document.querySelectorAll('.canvas-node').forEach(el => {
        el.addEventListener('click', (e) => {
            e.stopPropagation();
            selectNode(el.dataset.id);
        });
        el.addEventListener('mousedown', (e) => {
            if (e.target.closest('.connector-out, .connector-in')) return;
            e.stopPropagation();
            const id = el.dataset.id;
            const n = nodes[id];
            if (!n) return;
            isDragging = true;
            dragNodeId = id;
            dragOffsetX = e.clientX - (n.x + panX);
            dragOffsetY = e.clientY - (n.y + panY);
        });
    });
    // Connection points
    document.querySelectorAll('.connector-out').forEach(el => {
        el.addEventListener('mousedown', (e) => {
            e.stopPropagation();
            tempEdge = { source: el.dataset.id, mx: e.clientX, my: e.clientY };
        });
    });
    document.querySelectorAll('.connector-in').forEach(el => {
        el.addEventListener('mouseup', (e) => {
            if (tempEdge) {
                addEdge(tempEdge.source, el.dataset.id);
                tempEdge = null;
                render();
            }
        });
    });
}

// ── Canvas Mouse Events ──

function onCanvasMouseDown(e) {
    if (e.target === container || e.target.id === 'canvas-container' || e.target.id === 'no-graph-msg') {
        isPanning = true;
        panStartX = e.clientX - panX;
        panStartY = e.clientY - panY;
        selectedNodeId = null;
        updateProperties();
        render();
    }
}

function onCanvasMouseMove(e) {
    if (isDragging && dragNodeId) {
        const n = nodes[dragNodeId];
        if (n) {
            n.x = e.clientX - dragOffsetX;
            n.y = e.clientY - dragOffsetY;
            render();
        }
    } else if (isPanning) {
        panX = e.clientX - panStartX;
        panY = e.clientY - panStartY;
        render();
    }
    if (tempEdge) {
        tempEdge.mx = e.clientX;
        tempEdge.my = e.clientY;
        render();
    }
}

function onCanvasMouseUp(e) {
    isDragging = false;
    dragNodeId = null;
    isPanning = false;
    if (tempEdge && !e.target.closest('.connector-in')) {
        tempEdge = null;
        render();
    }
}

function onCanvasDblClick(e) {
    if (e.target === container || e.target.closest('.no-graph')) {
        const rect = container.getBoundingClientRect();
        const x = e.clientX - rect.left - panX - NODE_WIDTH / 2;
        const y = e.clientY - rect.top - panY - NODE_HEIGHT / 2;
        addNode('llm', x, y);
    }
}

// ── Drag from Palette ──

function onDragStart(e) {
    e.dataTransfer.setData('text/plain', e.target.dataset.type);
}

container.addEventListener('dragover', (e) => e.preventDefault());
container.addEventListener('drop', (e) => {
    e.preventDefault();
    const type = e.dataTransfer.getData('text/plain');
    if (!type) return;
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left - panX - NODE_WIDTH / 2;
    const y = e.clientY - rect.top - panY - NODE_HEIGHT / 2;
    addNode(type, x, y);
});

// ── Properties Panel ──

function updateProperties() {
    const el = document.getElementById('properties-content');
    if (!selectedNodeId || !nodes[selectedNodeId]) {
        el.innerHTML = '<div style="color:#94a3b8;font-size:13px;">Select a node to edit its properties</div>';
        return;
    }
    const n = nodes[selectedNodeId];
    let html = `<div class="prop-group"><label>Label</label><input type="text" value="${escapeHtml(n.label || '')}" onchange="updateNodeProp('${n.id}','label',this.value)"/></div>`;
    html += `<div class="prop-group"><label>Type</label><div style="font-size:13px;padding:4px 0;color:#64748b;">${n.type.toUpperCase()}</div></div>`;
    if (n.type === 'llm') {
        html += `<div class="prop-group"><label>Model</label><input type="text" value="${escapeHtml(n.model || '')}" onchange="updateNodeProp('${n.id}','model',this.value)"/></div>`;
        html += `<div class="prop-group"><label>Temperature</label><input type="number" step="0.1" min="0" max="2" value="${n.temperature || 0.7}" onchange="updateNodeProp('${n.id}','temperature',parseFloat(this.value))"/></div>`;
        html += `<div class="prop-group"><label>Max Tokens</label><input type="number" step="1" min="1" value="${n.max_tokens || 4096}" onchange="updateNodeProp('${n.id}','max_tokens',parseInt(this.value))"/></div>`;
        html += `<div class="prop-group"><label>System Prompt</label><textarea rows="3" style="width:100%;padding:6px 8px;border:1px solid var(--border-color,#e2e8f0);border-radius:4px;font-size:13px;font-family:monospace;" onchange="updateNodeProp('${n.id}','system_prompt',this.value)">${escapeHtml(n.system_prompt || '')}</textarea></div>`;
    }
    if (n.type === 'tool') {
        html += `<div class="prop-group"><label>Tool Name</label><input type="text" value="${escapeHtml(n.tool_name || '')}" onchange="updateNodeProp('${n.id}','tool_name',this.value)"/></div>`;
    }
    if (n.type === 'condition') {
        html += `<div class="prop-group"><label>Condition</label><input type="text" value="${escapeHtml(n.condition_expr || 'true')}" onchange="updateNodeProp('${n.id}','condition_expr',this.value)"/></div>`;
    }
    if (n.type === 'loop') {
        html += `<div class="prop-group"><label>Max Iterations</label><input type="number" min="1" value="${n.max_iterations || 10}" onchange="updateNodeProp('${n.id}','max_iterations',parseInt(this.value))"/></div>`;
    }
    el.innerHTML = html;
}

function updateNodeProp(id, key, value) {
    if (nodes[id]) {
        nodes[id][key] = value;
    }
}

// ── Graph Persistence ──

function getGraphData() {
    const graphNodes = {};
    Object.entries(nodes).forEach(([id, n]) => {
        graphNodes[id] = { type: n.type, label: n.label, x: n.x, y: n.y };
        if (n.model) graphNodes[id].model = n.model;
        if (n.temperature) graphNodes[id].temperature = n.temperature;
        if (n.max_tokens) graphNodes[id].max_tokens = n.max_tokens;
        if (n.system_prompt) graphNodes[id].system_prompt = n.system_prompt;
        if (n.tool_name) graphNodes[id].tool_name = n.tool_name;
        if (n.condition_expr) graphNodes[id].condition_expr = n.condition_expr;
        if (n.max_iterations) graphNodes[id].max_iterations = n.max_iterations;
    });
    const startNode = Object.keys(nodes).find(id => nodes[id].type === 'start') || Object.keys(nodes)[0] || '';
    return {
        id: 'canvas_' + Date.now().toString(36),
        name: 'Agent Graph ' + new Date().toLocaleDateString(),
        nodes: graphNodes,
        edges: edges.map(e => ({ source_id: e.source, target_id: e.target, label: e.label || '' })),
        start_node_id: startNode
    };
}

function loadGraphData(data) {
    nodes = {};
    edges = [];
    nextNodeId = 1;
    Object.entries(data.nodes || {}).forEach(([id, n]) => {
        const nodeId = 'node_' + (nextNodeId++);
        nodes[nodeId] = { id: nodeId, ...n };
        if (n.id && n.id !== nodeId) nodes[nodeId].id = nodeId;
    });
    edges = (data.edges || []).map(e => ({ source: e.source_id, target: e.target_id, label: e.label || '' }));
    if (Object.keys(nodes).length > 0) hideNoGraphMsg();
    else showNoGraphMsg();
    selectedNodeId = null;
    updateProperties();
    render();
}

async function saveGraph() {
    const data = getGraphData();
    const name = prompt('Graph name:', data.name);
    if (!name) return;
    data.name = name;
    try {
        const resp = await fetch('/admin/api/canvas/graphs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await resp.json();
        if (resp.ok) {
            data.id = result.id;
            refreshGraphList();
            alert('Graph saved: ' + result.id);
        } else {
            // Try update
            const updateResp = await fetch('/admin/api/canvas/graphs/' + data.id, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            if (updateResp.ok) {
                refreshGraphList();
                alert('Graph updated');
            } else {
                alert('Failed to save: ' + (await updateResp.json()).detail);
            }
        }
    } catch (e) {
        alert('Error saving graph: ' + e.message);
    }
}

async function loadGraph() {
    try {
        const resp = await fetch('/admin/api/canvas/graphs');
        const graphs = await resp.json();
        if (graphs.length === 0) { alert('No saved graphs'); return; }
        const sel = document.getElementById('graph-selector');
        sel.innerHTML = '<option value="">-- Select Graph --</option>';
        graphs.forEach(g => {
            sel.innerHTML += `<option value="${g.id}">${g.name || g.id}</option>`;
        });
        sel.style.display = 'inline-block';
        sel.focus();
    } catch (e) {
        alert('Error loading graphs: ' + e.message);
    }
}

async function onGraphSelect(graphId) {
    if (!graphId) return;
    document.getElementById('graph-selector').style.display = 'none';
    try {
        const resp = await fetch('/admin/api/canvas/graphs/' + graphId);
        if (!resp.ok) { alert('Graph not found'); return; }
        const data = await resp.json();
        loadGraphData(data);
    } catch (e) {
        alert('Error loading graph: ' + e.message);
    }
}

async function refreshGraphList() {
    try {
        const resp = await fetch('/admin/api/canvas/graphs');
        if (resp.ok) {
            const graphs = await resp.json();
            const sel = document.getElementById('graph-selector');
            sel.innerHTML = '<option value="">-- Select Graph --</option>';
            graphs.forEach(g => {
                sel.innerHTML += `<option value="${g.id}">${g.name || g.id}</option>`;
            });
        }
    } catch (e) { /* server may not be running */ }
}

function exportGraph() {
    const data = getGraphData();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (data.name || 'agent-graph').replace(/\s+/g, '_') + '.json';
    a.click();
    URL.revokeObjectURL(a.href);
}

function newGraph() {
    if (Object.keys(nodes).length > 0 && !confirm('Clear current graph?')) return;
    nodes = {};
    edges = [];
    nextNodeId = 1;
    selectedNodeId = null;
    panX = 0; panY = 0;
    addNode('start', 40, 200);
    addNode('llm', 260, 200);
    addNode('end', 480, 200);
    addEdge('node_1', 'node_2');
    addEdge('node_2', 'node_3');
    updateProperties();
}

async function runGraph() {
    const data = getGraphData();
    if (Object.keys(data.nodes).length === 0) { alert('Graph is empty'); return; }
    try {
        const resp = await fetch('/v1/agents/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ graph_id: data.id, graph: data, input: prompt('Enter input:') || '' })
        });
        if (!resp.ok) {
            const err = await resp.json();
            alert('Run failed: ' + (err.detail || resp.statusText));
            return;
        }
        const result = await resp.json();
        alert('Graph ready. Model: ' + result.model + '\nMessages prepared for /v1/chat/completions');
        console.log('Run result:', result);
    } catch (e) {
        alert('Error running graph: ' + e.message);
    }
}

function showNoGraphMsg() {
    document.getElementById('no-graph-msg').style.display = 'flex';
}

function hideNoGraphMsg() {
    document.getElementById('no-graph-msg').style.display = 'none';
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── Bootstrap ──
document.addEventListener('DOMContentLoaded', init);