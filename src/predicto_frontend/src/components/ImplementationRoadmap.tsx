import { Database, Brain, Rocket, Calendar, CheckCircle, ArrowRight } from "lucide-react";

const ImplementationRoadmap = () => {
  const phases = [
    {
      phase: "Phase 1",
      title: "Data Collection & Preparation",
      duration: "3-4 months",
      icon: Database,
      color: "text-primary",
      bgColor: "bg-primary/10",
      tasks: [
        "Collect & clean past POWERGRID project data",
        "Establish data pipelines and validation processes", 
        "Create standardized data formats",
        "Set up secure data infrastructure"
      ],
      deliverables: ["Clean historical dataset", "Data quality framework", "Security protocols"]
    },
    {
      phase: "Phase 2", 
      title: "Model Training & Validation",
      duration: "4-5 months",
      icon: Brain,
      color: "text-accent",
      bgColor: "bg-accent/10", 
      tasks: [
        "Train & validate forecasting models",
        "A/B testing with historical data",
        "Performance optimization and tuning",
        "Cross-validation with domain experts"
      ],
      deliverables: ["Validated ML models", "Performance benchmarks", "Accuracy metrics"]
    },
    {
      phase: "Phase 3",
      title: "Deployment & Integration", 
      duration: "2-3 months",
      icon: Rocket,
      color: "text-success",
      bgColor: "bg-success/10",
      tasks: [
        "Deploy dashboard + alerts for procurement teams",
        "Integration with existing POWERGRID systems",
        "User training and change management",
        "Monitoring and feedback loops"
      ],
      deliverables: ["Live dashboard", "Alert systems", "User documentation"]
    }
  ];

  const timeline = [
    { milestone: "Project Kickoff", month: "Month 1", status: "completed" },
    { milestone: "Data Infrastructure Ready", month: "Month 3", status: "completed" },
    { milestone: "First Model Training", month: "Month 6", status: "in-progress" },
    { milestone: "Pilot Testing", month: "Month 9", status: "upcoming" },
    { milestone: "Full Deployment", month: "Month 12", status: "upcoming" }
  ];

  return (
    <section className="py-24 bg-muted/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center space-x-2 bg-primary/10 backdrop-blur-sm border border-primary/20 rounded-full px-4 py-2 mb-6">
            <Calendar className="w-4 h-4 text-primary" />
            <span className="text-sm text-primary">Project Timeline</span>
          </div>
          <h2 className="text-4xl sm:text-5xl font-bold text-gradient mb-6">
            Implementation Roadmap
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            A systematic approach to deploying AI-powered forecasting across POWERGRID operations
          </p>
        </div>

        {/* Phases */}
        <div className="space-y-8 mb-16">
          {phases.map((phase, index) => (
            <div key={index} className="relative">
              <div className="glass-card p-8 animate-slide-up">
                <div className="flex flex-col lg:flex-row lg:items-start lg:space-x-8">
                  {/* Phase Header */}
                  <div className="flex items-center space-x-4 mb-6 lg:mb-0 lg:w-1/3">
                    <div className={`w-16 h-16 ${phase.bgColor} rounded-2xl flex items-center justify-center`}>
                      <phase.icon className={`w-8 h-8 ${phase.color}`} />
                    </div>
                    <div>
                      <div className="text-sm font-medium text-muted-foreground">{phase.phase}</div>
                      <h3 className="text-xl font-semibold text-foreground">{phase.title}</h3>
                      <div className="text-sm text-muted-foreground">{phase.duration}</div>
                    </div>
                  </div>

                  {/* Tasks & Deliverables */}
                  <div className="flex-1 grid md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="text-lg font-semibold text-foreground mb-3">Key Tasks</h4>
                      <ul className="space-y-2">
                        {phase.tasks.map((task, idx) => (
                          <li key={idx} className="flex items-start space-x-2 text-sm text-muted-foreground">
                            <CheckCircle className="w-4 h-4 text-success mt-0.5 flex-shrink-0" />
                            <span>{task}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h4 className="text-lg font-semibold text-foreground mb-3">Deliverables</h4>
                      <ul className="space-y-2">
                        {phase.deliverables.map((deliverable, idx) => (
                          <li key={idx} className="flex items-start space-x-2 text-sm text-muted-foreground">
                            <div className="w-2 h-2 bg-accent rounded-full mt-2 flex-shrink-0" />
                            <span>{deliverable}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* Connector Arrow */}
              {index < phases.length - 1 && (
                <div className="flex justify-center my-4">
                  <ArrowRight className="w-6 h-6 text-primary" />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Timeline */}
        <div className="glass-card p-8 animate-slide-up">
          <h3 className="text-2xl font-semibold text-foreground mb-8 text-center">Project Timeline</h3>
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            {timeline.map((item, index) => (
              <div key={index} className="flex items-center space-x-4">
                <div className={`w-4 h-4 rounded-full ${
                  item.status === 'completed' ? 'bg-success' :
                  item.status === 'in-progress' ? 'bg-accent animate-pulse' :
                  'bg-muted-foreground'
                }`} />
                <div className="text-center">
                  <div className="text-sm font-medium text-foreground">{item.milestone}</div>
                  <div className="text-xs text-muted-foreground">{item.month}</div>
                </div>
                {index < timeline.length - 1 && (
                  <ArrowRight className="w-4 h-4 text-muted-foreground hidden md:block ml-4" />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default ImplementationRoadmap;