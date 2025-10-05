import { MapPin, Zap, TrendingUp, Shield, Users, Building } from "lucide-react";

const NationalImpact = () => {
  const impacts = [
    {
      icon: Zap,
      title: "Faster Electrification of Rural Areas", 
      description: "Accelerated transmission line projects bring reliable electricity to remote regions",
      stats: "2M+ households benefited annually"
    },
    {
      icon: TrendingUp,
      title: "Reduced Cost Overruns",
      description: "Better forecasting prevents budget escalations in national grid expansion",
      stats: "₹500+ crores saved per year"
    },
    {
      icon: Shield,
      title: "Improved Supply Chain Resilience",
      description: "Enhanced material planning strengthens critical infrastructure development",
      stats: "30% faster project delivery"
    }
  ];

  const nationalStats = [
    { label: "Rural Areas Electrified Faster", value: "15-25%", icon: MapPin },
    { label: "Infrastructure Project Acceleration", value: "30%", icon: Building },
    { label: "National Grid Reliability", value: "99.8%", icon: Zap },
    { label: "Economic Impact", value: "₹2000+ Cr", icon: TrendingUp }
  ];

  const beneficiaries = [
    {
      group: "Rural Communities",
      benefit: "Faster access to reliable electricity",
      impact: "Improved quality of life and economic opportunities"
    },
    {
      group: "Industrial Sectors", 
      benefit: "Stable power supply for manufacturing",
      impact: "Enhanced productivity and competitiveness"
    },
    {
      group: "Government Initiatives",
      benefit: "Timely completion of electrification targets",
      impact: "Achievement of national development goals"
    }
  ];

  return (
    <section className="py-24 relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 hero-gradient opacity-5" />
      <div className="absolute top-20 left-10 w-32 h-32 bg-primary/10 rounded-full blur-3xl" />
      <div className="absolute bottom-20 right-10 w-40 h-40 bg-accent/10 rounded-full blur-3xl" />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center space-x-2 bg-success/10 backdrop-blur-sm border border-success/20 rounded-full px-4 py-2 mb-6">
            <MapPin className="w-4 h-4 text-success" />
            <span className="text-sm text-success">National Development</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-bold text-gradient mb-6">
            National Impact
          </h2>
          <p className="text-xl text-muted-foreground max-w-4xl mx-auto leading-relaxed">
            By minimizing material shortages, POWERGRID can complete projects faster, avoid crores in cost overruns, 
            and accelerate national infrastructure development.
          </p>
        </div>

        {/* Main Impact Areas */}
        <div className="grid lg:grid-cols-3 gap-8 mb-16">
          {impacts.map((impact, index) => (
            <div key={index} className="glass-card p-8 text-center hover-lift animate-slide-up">
              <div className="w-16 h-16 bg-gradient-primary rounded-2xl flex items-center justify-center mx-auto mb-6">
                <impact.icon className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-4">
                {impact.title}
              </h3>
              <p className="text-muted-foreground mb-4 leading-relaxed">
                {impact.description}
              </p>
              <div className="text-lg font-bold text-accent">
                {impact.stats}
              </div>
            </div>
          ))}
        </div>

        {/* National Statistics */}
        <div className="glass-card p-8 mb-16 animate-slide-up">
          <h3 className="text-2xl font-semibold text-foreground mb-8 text-center">
            Projected National Benefits
          </h3>
          <div className="grid md:grid-cols-4 gap-6">
            {nationalStats.map((stat, index) => (
              <div key={index} className="text-center p-4">
                <stat.icon className="w-8 h-8 text-accent mx-auto mb-4" />
                <div className="text-3xl font-bold text-gradient mb-2">{stat.value}</div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Beneficiary Impact */}
        <div className="glass-card p-8 animate-slide-up">
          <div className="flex items-center justify-center space-x-2 mb-8">
            <Users className="w-6 h-6 text-primary" />
            <h3 className="text-2xl font-semibold text-foreground">Stakeholder Benefits</h3>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            {beneficiaries.map((beneficiary, index) => (
              <div key={index} className="text-center p-6 bg-muted/20 rounded-xl">
                <h4 className="text-lg font-semibold text-foreground mb-3">
                  {beneficiary.group}
                </h4>
                <p className="text-sm text-primary font-medium mb-2">
                  {beneficiary.benefit}
                </p>
                <p className="text-sm text-muted-foreground">
                  {beneficiary.impact}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Call to Action */}
        <div className="text-center mt-16">
          <div className="glass-card p-8 inline-block">
            <h3 className="text-2xl font-semibold text-foreground mb-4">
              Transforming India's Infrastructure Future
            </h3>
            <p className="text-muted-foreground max-w-2xl">
              This AI-powered solution represents a significant step towards modernizing India's 
              infrastructure development, ensuring faster delivery and optimal resource utilization.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default NationalImpact;