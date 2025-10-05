import { Mail, Users, Github, Linkedin } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const ContactTeam = () => {
  return (
    <section id="contact" className="py-24 relative overflow-hidden">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12 animate-fade-in">
          <div className="inline-flex items-center space-x-2 bg-primary/10 backdrop-blur-sm border border-primary/20 rounded-full px-4 py-2 mb-6">
            <Users className="w-4 h-4 text-primary" />
            <span className="text-sm text-primary">Get in Touch</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-bold text-gradient mb-6">
            Contact Our Team
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Interested in collaborating on this prototype or learning more about the technology? 
            We're open to feedback and technical discussions.
          </p>
        </div>

        {/* Contact Card */}
        <Card className="glass-card p-8 animate-slide-up">
          <div className="text-center mb-8">
            <h3 className="text-2xl font-bold text-foreground mb-4">
              Team Predicto – SIH 2025
            </h3>
            <p className="text-muted-foreground mb-6">
              This prototype was developed as part of Smart India Hackathon 2025 to address 
              POWERGRID's materials demand forecasting challenge.
            </p>
          </div>

          {/* Contact Methods */}
          <div className="space-y-4 mb-8">
            <a
              href="mailto:team.predicto@example.com"
              className="flex items-center justify-center space-x-3 glass-card p-4 hover-lift transition-all"
            >
              <Mail className="w-5 h-5 text-primary" />
              <span className="text-foreground">team.predicto@example.com</span>
            </a>
            
            <div className="grid md:grid-cols-2 gap-4">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center space-x-3 glass-card p-4 hover-lift transition-all"
              >
                <Github className="w-5 h-5 text-foreground" />
                <span className="text-foreground">View Code</span>
              </a>
              
              <a
                href="https://linkedin.com"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center space-x-3 glass-card p-4 hover-lift transition-all"
              >
                <Linkedin className="w-5 h-5 text-foreground" />
                <span className="text-foreground">Connect</span>
              </a>
            </div>
          </div>

          {/* Info Box */}
          <div className="bg-muted/30 rounded-lg p-6 border border-border/50">
            <h4 className="font-semibold text-foreground mb-3 flex items-center">
              <span className="w-2 h-2 bg-success rounded-full mr-2 animate-pulse" />
              Available for Discussion
            </h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>✓ Technical implementation details and architecture</li>
              <li>✓ Integration with POWERGRID systems and data sources</li>
              <li>✓ Model training, validation, and performance optimization</li>
              <li>✓ Deployment strategies and security considerations</li>
            </ul>
          </div>
        </Card>

        {/* Disclaimer */}
        <div className="mt-8 text-center">
          <div className="inline-flex items-center space-x-2 bg-warning/10 backdrop-blur-sm border border-warning/20 rounded-lg px-4 py-3">
            <span className="text-sm text-muted-foreground">
              ⚠️ <strong>Prototype Disclaimer:</strong> This is a demonstration prototype built for SIH 2025. 
              Results shown are based on limited sample data for demonstration purposes only.
            </span>
          </div>
        </div>
      </div>

      {/* Background Elements */}
      <div className="absolute top-20 right-10 w-32 h-32 bg-primary/10 rounded-full blur-3xl animate-float" />
      <div className="absolute bottom-20 left-10 w-40 h-40 bg-accent/10 rounded-full blur-3xl animate-float" style={{ animationDelay: "3s" }} />
    </section>
  );
};

export default ContactTeam;
