function openSidebar() {
    document.getElementById("sidebar").classList.add("open");
    document.querySelector(".content").style.marginLeft = "270px";
}

function closeSidebar() {
    document.getElementById("sidebar").classList.remove("open");
    document.querySelector(".content").style.marginLeft = "20px";
}

function navigateToSection(sectionId) {
    closeSidebar(); // Close sidebar when navigating on mobile
    const headerHeight = document.getElementById("header").offsetHeight;
    const sectionTop = document.querySelector(sectionId).offsetTop;
    window.scrollTo({
        top: sectionTop - headerHeight + window.scrollY - 10, // Adjusted for additional spacing below header
        behavior: 'smooth'
    });
}

// Adjust header size on scroll
window.onscroll = function() {scrollFunction()};
function scrollFunction() {
    if (document.body.scrollTop > 80 || document.documentElement.scrollTop > 80) {
        document.getElementById("header").classList.add("shrink");
    } else {
        document.getElementById("header").classList.remove("shrink");
    }
}
