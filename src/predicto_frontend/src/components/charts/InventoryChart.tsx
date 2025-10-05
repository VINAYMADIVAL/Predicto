import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

interface InventoryChartProps {
  data?: Array<{
    month: string;
    stock: number;
    reorderPoint: number;
    demand: number;
  }>;
  type?: 'area' | 'pie';
  categoryData?: Array<{
    name: string;
    value: number;
    color: string;
  }>;
}

const defaultData = [
  { month: 'Jan', stock: 450, reorderPoint: 200, demand: 380 },
  { month: 'Feb', stock: 380, reorderPoint: 200, demand: 420 },
  { month: 'Mar', stock: 520, reorderPoint: 200, demand: 350 },
  { month: 'Apr', stock: 290, reorderPoint: 200, demand: 480 },
  { month: 'May', stock: 410, reorderPoint: 200, demand: 390 },
  { month: 'Jun', stock: 350, reorderPoint: 200, demand: 450 },
];

const defaultCategoryData = [
  { name: 'Raw Materials', value: 35, color: 'hsl(var(--primary))' },
  { name: 'Work in Progress', value: 25, color: 'hsl(var(--accent))' },
  { name: 'Finished Goods', value: 30, color: 'hsl(var(--success))' },
  { name: 'Safety Stock', value: 10, color: 'hsl(var(--warning))' },
];

const InventoryChart = ({ data = defaultData, type = 'area', categoryData = defaultCategoryData }: InventoryChartProps) => {
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background/95 backdrop-blur-sm border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm font-medium text-foreground mb-2">{`${label}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {`${entry.name}: ${entry.value}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const CustomPieTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      return (
        <div className="bg-background/95 backdrop-blur-sm border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm font-medium text-foreground">{data.name}</p>
          <p className="text-sm text-muted-foreground">{`${data.value}%`}</p>
        </div>
      );
    }
    return null;
  };

  if (type === 'pie') {
    return (
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={categoryData}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={100}
            paddingAngle={2}
            dataKey="value"
          >
            {categoryData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip content={<CustomPieTooltip />} />
        </PieChart>
      </ResponsiveContainer>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
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
        <Area
          type="monotone"
          dataKey="stock"
          name="Current Stock"
          stackId="1"
          stroke="hsl(var(--primary))"
          fill="hsl(var(--primary) / 0.3)"
          strokeWidth={2}
        />
        <Area
          type="monotone"
          dataKey="demand"
          name="Demand"
          stackId="2"
          stroke="hsl(var(--accent))"
          fill="hsl(var(--accent) / 0.3)"
          strokeWidth={2}
        />
        <Area
          type="monotone"
          dataKey="reorderPoint"
          name="Reorder Point"
          stroke="hsl(var(--warning))"
          fill="hsl(var(--warning) / 0.2)"
          strokeWidth={2}
          strokeDasharray="5 5"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};

export default InventoryChart;