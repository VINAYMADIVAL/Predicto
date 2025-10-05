import { Mail, Linkedin, Github } from "lucide-react";
import predictorLogo from "@/assets/predicto-logo.png";

const Footer = () => {
  const footerSections = [
    {
      title: "Prototype",
      links: [
        { name: "Challenge", href: "#challenge" },
        { name: "Solution", href: "#solution" },
        { name: "Demo", href: "#demo" },
        { name: "Roadmap", href: "#roadmap" },
      ],
    },
    {
      title: "Resources",
      links: [
        { name: "Documentation", href: "#" },
        { name: "GitHub Repository", href: "#" },
        { name: "Technical Paper", href: "#" },
      ],
    },
    {
      title: "SIH 2025",
      links: [
        { name: "About SIH", href: "https://www.sih.gov.in/" },
        { name: "POWERGRID Challenge", href: "#" },
        { name: "Team Info", href: "#contact" },
      ],
    },
  ];

  return (
    <footer className="bg-card border-t border-border/50 relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid lg:grid-cols-12 gap-8 mb-12">
          {/* Logo & Description */}
          <div className="lg:col-span-5 space-y-4">
            <div className="flex items-center space-x-3">
              <img src={predictorLogo} alt="Predicto" className="w-10 h-10" />
              <div className="flex flex-col">
                <span className="text-xl font-bold text-gradient">Predicto</span>
                <span className="text-xs text-muted-foreground">SIH 2025 Prototype</span>
              </div>
            </div>
            <p className="text-muted-foreground text-sm leading-relaxed max-w-xs">
              AI-powered materials demand forecasting for POWERGRID infrastructure projects. 
              Built under Smart India Hackathon 2025.
            </p>
            <div className="flex space-x-4">
              {[
                { icon: Linkedin, href: "#" },
                { icon: Github, href: "#" },
                { icon: Mail, href: "mailto:team.predicto@example.com" },
              ].map((social, index) => (
                <a
                  key={index}
                  href={social.href}
                  className="w-9 h-9 bg-muted/50 rounded-lg flex items-center justify-center hover:bg-primary hover:text-primary-foreground transition-all hover-lift"
                >
                  <social.icon className="w-4 h-4" />
                </a>
              ))}
            </div>
          </div>

          {/* Links Grid */}
          <div className="lg:col-span-5 grid grid-cols-3 gap-8">
            {footerSections.map((section, index) => (
              <div key={index}>
                <h3 className="text-sm font-semibold text-foreground mb-4">
                  {section.title}
                </h3>
                <ul className="space-y-3">
                  {section.links.map((link, linkIndex) => (
                    <li key={linkIndex}>
                      <a
                        href={link.href}
                        className="text-muted-foreground hover:text-primary transition-colors text-sm"
                      >
                        {link.name}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          {/* Disclaimer */}
          <div className="lg:col-span-2 space-y-4">
            <h3 className="text-sm font-semibold text-foreground mb-4">Disclaimer</h3>
            <div className="bg-warning/10 border border-warning/20 rounded-lg p-4">
              <p className="text-xs text-muted-foreground leading-relaxed">
                ⚠️ This is a prototype developed for Smart India Hackathon 2025. 
                Results and forecasts shown are based on limited sample data and are 
                for demonstration purposes only. Not intended for production use without 
                further validation with real POWERGRID data.
              </p>
            </div>
            <div className="flex items-center space-x-3 text-sm">
              <Mail className="w-4 h-4 flex-shrink-0 text-muted-foreground" />
              <span className="text-muted-foreground text-xs">team.predicto@example.com</span>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-border/50 pt-8 flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
          <p className="text-sm text-muted-foreground">
            © {new Date().getFullYear()} Predicto – SIH 2025 Prototype. Built for educational and demonstration purposes.
          </p>
          <div className="flex items-center space-x-2 text-sm text-muted-foreground">
            <span>Powered by</span>
            <span className="text-primary font-semibold">Machine Learning</span>
          </div>
        </div>
      </div>

      {/* Background Elements */}
      <div className="absolute bottom-0 left-0 w-96 h-96 bg-primary/5 rounded-full blur-3xl -translate-x-48 translate-y-48" />
      <div className="absolute top-0 right-0 w-96 h-96 bg-accent/5 rounded-full blur-3xl translate-x-48 -translate-y-48" />
    </footer>
  );
};

export default Footer;
