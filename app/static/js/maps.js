/* Leaflet Maps */
(function () {
  const DEFAULT_CENTER = [30.0444, 31.2357]; // Cairo fallback
  const DEFAULT_ZOOM = 12;

  let mainMap = null;
  let mainMarkersLayer = null;

  function ensureLeaflet() {
    if (typeof L === 'undefined') {
      console.error('Leaflet not loaded. Include Leaflet CSS/JS before maps.js');
      return false;
    }
    return true;
  }

  function googleMapsLink(lat, lon) {
    if (typeof lat === 'number' && typeof lon === 'number') {
      return `https://www.google.com/maps?q=${lat},${lon}`;
    }
    return '#';
  }

  function initMainMap() {
    if (!ensureLeaflet()) return;
    const container = document.getElementById('mainMap');
    if (!container) return;
    if (mainMap) return; // already initialized

    mainMap = L.map('mainMap', {
      zoomControl: true,
      attributionControl: false,
      dragging: true,
      scrollWheelZoom: true,
      doubleClickZoom: true,
      boxZoom: true,
      keyboard: true
    }).setView(DEFAULT_CENTER, DEFAULT_ZOOM);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      subdomains: 'abc',
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(mainMap);

    mainMarkersLayer = L.layerGroup().addTo(mainMap);
  }

  function updateMainMap(restaurants, userLocation) {
    if (!ensureLeaflet()) return;
    initMainMap();
    if (!mainMap || !mainMarkersLayer) return;

    mainMarkersLayer.clearLayers();
    const bounds = [];

    if (Array.isArray(restaurants)) {
      restaurants.forEach(r => {
        const lat = r.latitude ?? r.lat;
        const lon = r.longitude ?? r.lon;
        if (typeof lat === 'number' && typeof lon === 'number') {
          const marker = L.marker([lat, lon]).addTo(mainMarkersLayer);
          const name = r.name || r.restaurant_name || 'Unknown';
          const addr = r.address || '';
          const link = r.link || googleMapsLink(lat, lon);
          const popupHtml = `
            <div style="text-align:center; min-width:220px;">
              <strong style="font-size:1.05rem;">${escapeHtml(name)}</strong><br/>
              <small style="color:#666;">${escapeHtml(addr)}</small><br/>
              <a href="${link}" target="_blank" style="display:inline-block; margin-top:6px; color:#334; text-decoration:none; border:1px solid #ccd; border-radius:6px; padding:4px 8px;">Open in Google Maps</a>
            </div>`;
          marker.bindPopup(popupHtml);
          bounds.push([lat, lon]);
        }
      });
    }

    if (userLocation && typeof userLocation.latitude === 'number' && typeof userLocation.longitude === 'number') {
      const u = L.circleMarker([userLocation.latitude, userLocation.longitude], {
        radius: 10,
        color: '#2b6cb0',
        fillColor: '#3182ce',
        fillOpacity: 0.9
      }).addTo(mainMarkersLayer);
      u.bindPopup('<div style="min-width:160px;">You are here</div>');
      bounds.push([userLocation.latitude, userLocation.longitude]);
    }

    if (bounds.length > 0) {
      mainMap.fitBounds(bounds, { padding: [40, 40] });
    }
  }

  function renderMiniMap(containerId, restaurant, userLocation) {
    if (!ensureLeaflet()) return;
    const el = document.getElementById(containerId);
    if (!el) return;

    // Avoid re-initializing an existing map
    if (el._leaflet_id) {
      return;
    }

    const lat = restaurant.latitude ?? restaurant.lat;
    const lon = restaurant.longitude ?? restaurant.lon;
    if (typeof lat !== 'number' || typeof lon !== 'number') return;

    const m = L.map(containerId, {
      zoomControl: true,
      attributionControl: false,
      dragging: true,
      scrollWheelZoom: true,
      doubleClickZoom: true,
      boxZoom: true,
      keyboard: true
    }).setView([lat, lon], 15);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      subdomains: 'abc',
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(m);

    const name = restaurant.name || restaurant.restaurant_name || 'Unknown';
    const addr = restaurant.address || '';
    const link = restaurant.link || googleMapsLink(lat, lon);

    const marker = L.marker([lat, lon]).addTo(m);
    marker.bindPopup(`
      <div style="text-align:center; min-width:220px; padding:6px 8px; border-radius:8px; border:1px solid #e6eaf0; box-shadow:0 4px 10px rgba(0,0,0,0.06);">
        <strong style="display:block; margin-bottom:4px;">${escapeHtml(name)}</strong>
        <small style="color:#666;">${escapeHtml(addr)}</small><br/>
        <a href="${link}" target="_blank" style="display:inline-block; margin-top:6px; color:#334; text-decoration:none; border:1px solid #ccd; border-radius:16px; padding:4px 10px; background:#fff;">🗺️ Open in Google Maps</a>
      </div>
    `);

    // Optionally show user marker on mini map
    if (userLocation && typeof userLocation.latitude === 'number' && typeof userLocation.longitude === 'number') {
      L.circleMarker([userLocation.latitude, userLocation.longitude], {
        radius: 6,
        color: '#2b6cb0',
        fillColor: '#3182ce',
        fillOpacity: 0.9
      }).addTo(m).bindPopup('<div style="min-width:140px; padding:4px 6px; border:1px solid #e6eaf0; border-radius:6px;">📍 You are here</div>');
      const bounds = L.latLngBounds([[lat, lon], [userLocation.latitude, userLocation.longitude]]);
      m.fitBounds(bounds, { padding: [20, 20] });
    }
  }

  function renderAllMiniMaps(restaurants, userLocation) {
    if (!Array.isArray(restaurants)) return;
    restaurants.forEach((r, idx) => {
      const placeId = r.place_id || `place-${idx}`;
      renderMiniMap(`map-${placeId}`, r, userLocation);
    });
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
  }

  window.Maps = {
    initMainMap,
    updateMainMap,
    renderMiniMap,
    renderAllMiniMaps
  };
})();


