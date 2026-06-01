/**
 * LexisSearch — Frontend JavaScript
 *
 * Handles:
 * - Dark / light mode toggle
 * - Search bar autocomplete
 * - Ranking selector UI
 * - Loading spinner on search submit
 * - Navbar scroll effect
 * - Smooth interaction polish
 */

// ── Dark / Light Mode ─────────────────────────────────────────
(function initTheme() {
  // Check if user had a preference saved
  const saved = localStorage.getItem('lexis-theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
  updateThemeIcon(saved);
})();

function updateThemeIcon(theme) {
  const icon = document.getElementById('themeIcon');
  if (!icon) return;
  icon.className = theme === 'dark' ? 'bi bi-sun' : 'bi bi-moon-stars';
}

document.addEventListener('DOMContentLoaded', () => {

  // ── Theme toggle ────────────────────────────────────────────
  const themeBtn = document.getElementById('themeToggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('lexis-theme', next);
      updateThemeIcon(next);
    });
  }

  // ── Navbar shadow on scroll ──────────────────────────────────
  const nav = document.getElementById('main-nav');
  if (nav) {
    window.addEventListener('scroll', () => {
      nav.classList.toggle('scrolled', window.scrollY > 10);
    }, { passive: true });
  }

  // ── Main search form: loading spinner ──────────────────────
  const mainForm = document.getElementById('main-search-form');
  if (mainForm) {
    mainForm.addEventListener('submit', () => {
      const btn = document.getElementById('search-btn');
      if (btn) btn.classList.add('loading');
    });
  }

  const navForm = document.getElementById('nav-search-form');
  if (navForm) {
    navForm.addEventListener('submit', () => {
      const btn = navForm.querySelector('button[type=submit]');
      if (btn) btn.disabled = true;
    });
  }

  // ── Autocomplete ────────────────────────────────────────────
  const searchInput = document.getElementById('main-search-input');
  const dropdown    = document.getElementById('autocomplete-dropdown');

  if (searchInput && dropdown) {
    let debounceTimer = null;

    searchInput.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      const q = searchInput.value.trim();
      if (q.length < 2) {
        hideDropdown();
        return;
      }
      // Debounce — wait 200ms after typing stops before fetching
      debounceTimer = setTimeout(() => fetchSuggestions(q), 200);
    });

    // Hide dropdown when clicking outside
    document.addEventListener('click', (e) => {
      if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {
        hideDropdown();
      }
    });

    // Keyboard navigation inside the dropdown
    searchInput.addEventListener('keydown', (e) => {
      const items = dropdown.querySelectorAll('.autocomplete-item');
      const active = dropdown.querySelector('.autocomplete-item.selected');
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (!active) {
          items[0]?.classList.add('selected');
        } else {
          const next = active.nextElementSibling;
          active.classList.remove('selected');
          if (next) next.classList.add('selected');
        }
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (active) {
          const prev = active.previousElementSibling;
          active.classList.remove('selected');
          if (prev) prev.classList.add('selected');
        }
      } else if (e.key === 'Enter' && active) {
        searchInput.value = active.dataset.value;
        hideDropdown();
        // Let the form submit normally
      } else if (e.key === 'Escape') {
        hideDropdown();
      }
    });
  }

  function fetchSuggestions(q) {
    fetch(`/autocomplete?q=${encodeURIComponent(q)}`)
      .then(r => r.json())
      .then(data => {
        if (data.suggestions && data.suggestions.length > 0) {
          renderDropdown(data.suggestions);
        } else {
          hideDropdown();
        }
      })
      .catch(() => hideDropdown());
  }

  function renderDropdown(suggestions) {
    dropdown.innerHTML = '';
    suggestions.forEach(s => {
      const item = document.createElement('div');
      item.className = 'autocomplete-item';
      item.dataset.value = s;
      item.innerHTML = `<i class="bi bi-search"></i> ${escapeHtml(s)}`;
      item.addEventListener('mousedown', (e) => {
        e.preventDefault(); // Prevent input blur before click registers
        searchInput.value = s;
        hideDropdown();
        // Auto-submit
        document.getElementById('main-search-form')?.submit();
      });
      dropdown.appendChild(item);
    });
    dropdown.classList.add('show');
  }

  function hideDropdown() {
    if (dropdown) {
      dropdown.classList.remove('show');
      dropdown.innerHTML = '';
    }
  }

  // ── Ranking selector (home page) ────────────────────────────
  // Handled inline via onclick="selectRanking(this)" but we also
  // animate the active state here for any rank-btn clicks.
  document.querySelectorAll('.rank-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      // Highlight is handled by the onclick functions in the templates
      // We add a little pulse to give feedback
      btn.style.transform = 'scale(0.95)';
      setTimeout(() => { btn.style.transform = ''; }, 120);
    });
  });

  // ── Stagger result card animations ──────────────────────────
  // Each card has animation-delay set via style in the template,
  // but we also trigger a slight entrance effect here.
  const cards = document.querySelectorAll('.result-card');
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.05 });
    cards.forEach(card => observer.observe(card));
  }

});

// ── Ranking selector helpers (called from templates) ──────────
function selectRanking(btn) {
  document.querySelectorAll('#main-search-form .rank-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('ranking-input').value = btn.dataset.value;
}

// Utility: escape HTML to prevent injection in autocomplete items
function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
