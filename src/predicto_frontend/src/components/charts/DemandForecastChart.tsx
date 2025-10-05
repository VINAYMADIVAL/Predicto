import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

interface DemandForecastChartProps {
  data: Array<{
    month: string;
    predicted: number;
    actual: number;
  }>;
  type?: 'line' | 'bar';
}

const DemandForecastChart = ({ data, type = 'line' }: DemandForecastChartProps) => {
  const chartData = data.map(item => ({
    ...item,
    accuracy: Math.round(((1 - Math.abs(item.predicted - item.actual) / item.predicted) * 100))
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background/95 backdrop-blur-sm border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm font-medium text-foreground mb-2">{`${label}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {`${entry.name}: ${entry.value}${entry.dataKey === 'accuracy' ? '%' : ''}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  if (type === 'bar') {
    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis 
            dataKey="month" 
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
          />
          <YAxis 
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Bar 
            dataKey="predicted" 
            name="Predicted"
            fill="hsl(var(--primary))"
            radius={[4, 4, 0, 0]}
          />
          <Bar 
            dataKey="actual" 
            name="Actual"
            fill="hsl(var(--accent))"
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
        <XAxis 
          dataKey="month" 
          stroke="hsl(var(--muted-foreground))"
          fontSize={12}
        />
        <YAxis 
          stroke="hsl(var(--muted-foreground))"
          fontSize={12}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        <Line 
          type="monotone" 
          dataKey="predicted" 
          name="Predicted"
          stroke="hsl(var(--primary))" 
          strokeWidth={3}
          dot={{ fill: 'hsl(var(--primary))', strokeWidth: 2, r: 4 }}
          activeDot={{ r: 6, fill: 'hsl(var(--primary))' }}
        />
        <Line 
          type="monotone" 
          dataKey="actual" 
          name="Actual"
          stroke="hsl(var(--accent))" 
          strokeWidth={3}
          dot={{ fill: 'hsl(var(--accent))', strokeWidth: 2, r: 4 }}
          activeDot={{ r: 6, fill: 'hsl(var(--accent))' }}
        />
        <Line 
          type="monotone" 
          dataKey="accuracy" 
          name="Accuracy %"
          stroke="hsl(var(--success))" 
          strokeWidth={2}
          strokeDasharray="5 5"
          dot={{ fill: 'hsl(var(--success))', strokeWidth: 1, r: 3 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default DemandForecastChart;