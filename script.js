
"use strict";

/* =========================================================
   CardioShield-AgeLess
   File: script.js
   Homepage interactions
   ========================================================= */

document.addEventListener("DOMContentLoaded", function () {
    initializeCopyrightYear();
    initializeMobileNavigation();
    initializeSmoothScrolling();
    initializeHeartImageFallback();
});


/* 1. AUTOMATIC COPYRIGHT YEAR */

function initializeCopyrightYear() {
    const yearElement = document.getElementById("currentYear");

    if (yearElement) {
        yearElement.textContent = new Date().getFullYear();
    }
}


/* 2. MOBILE NAVIGATION */

function initializeMobileNavigation() {
    const menuToggle = document.getElementById("menuToggle");
    const navLinks = document.getElementById("navLinks");

    if (!menuToggle || !navLinks) {
        return;
    }

    function closeMenu() {
        navLinks.classList.remove("nav-open");
        menuToggle.setAttribute("aria-expanded", "false");
    }

    function toggleMenu() {
        const isOpen = navLinks.classList.toggle("nav-open");

        menuToggle.setAttribute("aria-expanded", String(isOpen));
    }

    menuToggle.addEventListener("click", toggleMenu);

    // Close the mobile menu after a navigation link is selected.
    navLinks.querySelectorAll("a").forEach(function (link) {
        link.addEventListener("click", closeMenu);
    });

    // Close the menu when the Escape key is pressed.
    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape") {
            closeMenu();
            menuToggle.focus();
        }
    });

    // Close the mobile menu when clicking outside the navigation.
    document.addEventListener("click", function (event) {
        const clickedInsideMenu = navLinks.contains(event.target);
        const clickedToggle = menuToggle.contains(event.target);

        if (!clickedInsideMenu && !clickedToggle) {
            closeMenu();
        }
    });

    // Reset mobile navigation when returning to desktop width.
    window.addEventListener("resize", function () {
        if (window.innerWidth > 760) {
            closeMenu();
        }
    });
}


/* 3. SMOOTH INTERNAL NAVIGATION */

function initializeSmoothScrolling() {
    const internalLinks = document.querySelectorAll('a[href^="#"]');

    internalLinks.forEach(function (link) {
        link.addEventListener("click", function (event) {
            const targetId = link.getAttribute("href");

            if (!targetId || targetId === "#") {
                return;
            }

            let target;

            try {
                target = document.querySelector(targetId);
            } catch (error) {
                // Ignore malformed fragment selectors.
                return;
            }

            if (!target) {
                return;
            }

            event.preventDefault();

            target.scrollIntoView({
                behavior: window.matchMedia(
                    "(prefers-reduced-motion: reduce)"
                ).matches ? "auto" : "smooth",
                block: "start"
            });

            // Keep the browser's address bar aligned with the section.
            if (window.location.hash !== targetId) {
                window.history.replaceState(null, "", targetId);
            }
        });
    });
}


/* 4. HEART IMAGE FALLBACK */

function initializeHeartImageFallback() {
    const heartImage = document.getElementById("heartImage");
    const imageError = document.getElementById("imageError");

    if (!heartImage || !imageError) {
        return;
    }

    function showImageError() {
        heartImage.hidden = true;
        imageError.hidden = false;
    }

    function showHeartImage() {
        heartImage.hidden = false;
        imageError.hidden = true;
    }

    heartImage.addEventListener("error", showImageError);
    heartImage.addEventListener("load", showHeartImage);

    // Handle images that have already loaded or failed
    // before the event listeners were attached.
    if (heartImage.complete) {
        if (heartImage.naturalWidth === 0) {
            showImageError();
        } else {
            showHeartImage();
        }
    }
}
