import { Server, Shield, Lock, Database, CheckCircle, Globe } from "lucide-react";

const DeploymentSecurity = () => {
  const deploymentFeatures = [
    {
      icon: Server,
      title: "On-Premise Deployment",
      description: "Complete solution deployment on POWERGRID's own servers",
      benefits: ["Full data sovereignty", "No external dependencies", "Custom configurations"]
    },
    {
      icon: Database,
      title: "Data Sovereignty",
      description: "All sensitive project data remains within POWERGRID infrastructure", 
      benefits: ["Zero data leakage", "Compliance with regulations", "Complete control"]
    },
    {
      icon: Shield,
      title: "Enterprise Security",
      description: "Military-grade security protocols and access controls",
      benefits: ["Role-based permissions", "Audit trails", "Encrypted communications"]
    }
  ];

  const securityCompliance = [
    {
      standard: "SOC 2 Type II",
      description: "Security, availability, and confidentiality controls",
      status: "Compliant"
    },
    {
      standard: "ISO 27001",
      description: "Information security management systems",
      status: "Certified" 
    },
    {
      standard: "GDPR",
      description: "Data protection and privacy regulations",
      status: "Compliant"
    },
    {
      standard: "Government of India IT Policy",
      description: "National cybersecurity framework compliance",
      status: "Aligned"
    }
  ];

  const architectureComponents = [
    { component: "Load Balancer", purpose: "High availability and traffic distribution" },
    { component: "Application Servers", purpose: "Scalable processing and model execution" },
    { component: "Database Cluster", purpose: "Redundant data storage and backup" },
    { component: "Security Gateway", purpose: "Firewall and intrusion detection" },
    { component: "Monitoring Suite", purpose: "Real-time performance and security monitoring" }
  ];

  return (
    <section className="py-24 bg-muted/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center space-x-2 bg-primary/10 backdrop-blur-sm border border-primary/20 rounded-full px-4 py-2 mb-6">
            <Shield className="w-4 h-4 text-primary" />
            <span className="text-sm text-primary">Security & Deployment</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-bold text-gradient mb-6">
            Deployment & Security
          </h2>
        </div>

        {/* Key Deployment Features */}
        <div className="grid lg:grid-cols-3 gap-8 mb-16">
          {deploymentFeatures.map((feature, index) => (
            <div key={index} className="glass-card p-8 hover-lift animate-slide-up">
              <div className="w-16 h-16 bg-gradient-primary rounded-2xl flex items-center justify-center mb-6">
                <feature.icon className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-4">
                {feature.title}
              </h3>
              <p className="text-muted-foreground mb-6">
                {feature.description}
              </p>
              <ul className="space-y-2">
                {feature.benefits.map((benefit, idx) => (
                  <li key={idx} className="flex items-center space-x-2 text-sm text-muted-foreground">
                    <CheckCircle className="w-4 h-4 text-success flex-shrink-0" />
                    <span>{benefit}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Security Compliance */}
        <div className="glass-card p-8 mb-16 animate-slide-up">
          <div className="flex items-center justify-center space-x-2 mb-8">
            <Lock className="w-6 h-6 text-accent" />
            <h3 className="text-2xl font-semibold text-foreground">Security Compliance</h3>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {securityCompliance.map((item, index) => (
              <div key={index} className="text-center p-6 bg-muted/20 rounded-xl">
                <div className="w-12 h-12 bg-success/20 rounded-full flex items-center justify-center mx-auto mb-4">
                  <CheckCircle className="w-6 h-6 text-success" />
                </div>
                <h4 className="text-lg font-semibold text-foreground mb-2">{item.standard}</h4>
                <p className="text-sm text-muted-foreground mb-3">{item.description}</p>
                <span className="inline-block px-3 py-1 bg-success/10 text-success text-xs font-medium rounded-full">
                  {item.status}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Architecture Overview */}
        <div className="glass-card p-8 animate-slide-up">
          <div className="flex items-center justify-center space-x-2 mb-8">
            <Globe className="w-6 h-6 text-primary" />
            <h3 className="text-2xl font-semibold text-foreground">System Architecture</h3>
          </div>
          <div className="space-y-4">
            {architectureComponents.map((component, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-muted/20 rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className="w-3 h-3 bg-primary rounded-full" />
                  <span className="font-medium text-foreground">{component.component}</span>
                </div>
                <span className="text-muted-foreground text-sm">{component.purpose}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Key Benefits */}
        <div className="mt-16 text-center">
          <div className="glass-card p-8 inline-block max-w-4xl">
            <h3 className="text-2xl font-semibold text-foreground mb-4">
              Enterprise-Ready Solution
            </h3>
            <div className="grid md:grid-cols-2 gap-6 text-left">
              <div>
                <h4 className="font-semibold text-primary mb-2">On-Premise Benefits:</h4>
                <p className="text-muted-foreground text-sm">
                  Solution can be deployed on POWERGRID's on-premise servers to ensure data sovereignty 
                  and complete control over sensitive infrastructure data.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-primary mb-2">Security Standards:</h4>
                <p className="text-muted-foreground text-sm">
                  Compliant with enterprise-grade security standards (SOC 2, ISO 27001) ensuring 
                  the highest levels of data protection and system reliability.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DeploymentSecurity;