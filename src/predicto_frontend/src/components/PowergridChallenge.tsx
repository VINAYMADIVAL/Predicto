import { Zap, AlertTriangle, TrendingDown } from "lucide-react";

const PowergridChallenge = () => {
  const challenges = [
    {
      icon: AlertTriangle,
      title: "Project Delays",
      description: "Material shortages cause costly delays in tower and substation construction"
    },
    {
      icon: TrendingDown,
      title: "Cost Overruns", 
      description: "Poor forecasting leads to emergency procurement at premium prices"
    },
    {
      icon: Zap,
      title: "Infrastructure Impact",
      description: "Delays affect national grid expansion and rural electrification goals"
    }
  ];

  return (
    <section className="py-24 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center space-x-2 bg-warning/10 backdrop-blur-sm border border-warning/20 rounded-full px-4 py-2 mb-6">
            <AlertTriangle className="w-4 h-4 text-warning" />
            <span className="text-sm text-warning">Critical Challenge</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-bold text-gradient mb-6">
            The POWERGRID Challenge
          </h2>
          <p className="text-xl text-muted-foreground max-w-4xl mx-auto leading-relaxed">
            POWERGRID executes large infrastructure projects (towers, substations) where delays must be avoided. 
            Materials demand forecasting is critical to prevent shortages, overruns, and delays.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 animate-slide-up">
          {challenges.map((challenge, index) => (
            <div key={index} className="glass-card p-8 text-center hover-lift">
              <div className="w-16 h-16 bg-gradient-primary rounded-2xl flex items-center justify-center mx-auto mb-6">
                <challenge.icon className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-4">
                {challenge.title}
              </h3>
              <p className="text-muted-foreground">
                {challenge.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default PowergridChallenge;