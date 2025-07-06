import React, { useState, useEffect, useMemo, useCallback } from "react";
import ReactECharts from "echarts-for-react";
import {
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  Button,
  Alert,
  Stack,
  ToggleButtonGroup,
  ToggleButton,
  Paper,
  Checkbox,
  TextField,
  Collapse,
  Autocomplete,
  Chip,
} from "@mui/material";
// --- ⬇️ IMPORT ECHARTS TYPE ---
import type { ECharts } from "echarts";

// Icons
import SortByAlphaIcon from '@mui/icons-material/SortByAlpha';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import TuneIcon from '@mui/icons-material/Tune';
import CloseIcon from '@mui/icons-material/Close';
import CheckBoxOutlineBlankIcon from '@mui/icons-material/CheckBoxOutlineBlank';
import CheckBoxIcon from '@mui/icons-material/CheckBox';
import FunctionsIcon from '@mui/icons-material/Functions';

// Helper for deep cloning ECharts options
function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== "object") return obj;
  if (obj instanceof Date) return new Date(obj.getTime()) as any;
  if (Array.isArray(obj)) return obj.map(deepClone) as any;
  const cloned = {} as T;
  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      cloned[key] = deepClone(obj[key]);
    }
  }
  return cloned;
}

const SUPPORTED_CHART_TYPES_FOR_SWITCHING = [
  'bar', 'line', 'area', 'pie', 'scatter', 'radar', 'funnel', 'histogram'
] as const;
type SwitchableChartType = typeof SUPPORTED_CHART_TYPES_FOR_SWITCHING[number];

interface NumericalFilterConfig {
  condition: 'none' | '>' | '<' | '>=' | '<=' | '==';
  value: string;
  isActive: boolean;
}

interface ChartTabProps {
  chartOption: any;
  onChartRendered: (chartInstance: ECharts) => void; // --- ⬇️ ADD THIS PROP ---
}

export function ChartTab({ chartOption: chartOptionFromProp, onChartRendered }: ChartTabProps) { // --- ⬇️ ACCEPT PROP HERE ---
  const [initialLLMOption, setInitialLLMOption] = useState<any | null>(null);
  const [llmChartType, setLlmChartType] = useState<SwitchableChartType | undefined>(undefined);
  const [selectedChartType, setSelectedChartType] = useState<SwitchableChartType | undefined>(undefined);
  const [compatibleChartTypes, setCompatibleChartTypes] = useState<SwitchableChartType[]>([]);
  const [transformationError, setTransformationError] = useState<string | null>(null);

  const [sortConfig, setSortConfig] = useState<{ key: 'name' | 'value'; order: 'asc' | 'desc' | 'none' }>({ key: 'value', order: 'none' });
  const [availableFilterCategories, setAvailableFilterCategories] = useState<string[]>([]);
  const [selectedFilterCategories, setSelectedFilterCategories] = useState<Set<string>>(new Set());

  const [numericalFilter, setNumericalFilter] = useState<NumericalFilterConfig>({
    condition: 'none', value: '', isActive: false,
  });
  const [isNumericalFilterPossible, setIsNumericalFilterPossible] = useState(false);

  const [controlsOpen, setControlsOpen] = useState(false);

  // ... (all existing useEffects and handlers remain unchanged) ...
  useEffect(() => {
    if (chartOptionFromProp && typeof chartOptionFromProp === 'object') {
      const pristineOption = deepClone(chartOptionFromProp);
      setInitialLLMOption(pristineOption);
      setTransformationError(null);
      setSortConfig({ key: 'value', order: 'none' });
      setNumericalFilter({ condition: 'none', value: '', isActive: false });

      const seriesArray = pristineOption.series ? (Array.isArray(pristineOption.series) ? pristineOption.series : [pristineOption.series]) : [];
      let typeFromLLM: SwitchableChartType | undefined = undefined;

      if (seriesArray.length > 0 && seriesArray[0].type) {
        const rawType = seriesArray[0].type.toLowerCase();
        if (SUPPORTED_CHART_TYPES_FOR_SWITCHING.includes(rawType as SwitchableChartType)) {
          typeFromLLM = rawType as SwitchableChartType;
        }
      }

      const fallbackType: SwitchableChartType = 'bar';
      const determinedLlmType = typeFromLLM || fallbackType;

      setLlmChartType(determinedLlmType);
      setSelectedChartType(determinedLlmType);

      let CTypes: SwitchableChartType[] = [determinedLlmType];
      const firstSeriesData = seriesArray.length > 0 ? seriesArray[0].data : null;
      const xAxis = pristineOption.xAxis?.[0] || pristineOption.xAxis;
      const xAxisDataExists = xAxis?.data && Array.isArray(xAxis.data);

      if (['bar', 'line', 'area', 'histogram', 'scatter'].includes(determinedLlmType)) {
        CTypes.push('bar', 'line', 'area', 'histogram');
        if (determinedLlmType === 'scatter') CTypes.push('scatter'); else CTypes.push('scatter');
        if (xAxisDataExists && firstSeriesData) CTypes.push('pie', 'funnel', 'radar');
      } else if (['pie', 'funnel'].includes(determinedLlmType)) {
        CTypes.push('pie', 'funnel');
        if (firstSeriesData?.every((d: any) => d.name !== undefined && d.value !== undefined)) {
          CTypes.push('bar', 'line', 'area');
        }
      } else if (determinedLlmType === 'radar') {
        CTypes.push('radar', 'line', 'bar');
      }
      setCompatibleChartTypes([...new Set(CTypes.filter(t => SUPPORTED_CHART_TYPES_FOR_SWITCHING.includes(t)))]);

    } else {
        setInitialLLMOption(null); setLlmChartType(undefined); setSelectedChartType(undefined);
        setCompatibleChartTypes([]); setTransformationError(null);
        setSortConfig({ key: 'value', order: 'none' });
        setAvailableFilterCategories([]); setSelectedFilterCategories(new Set());
        setNumericalFilter({ condition: 'none', value: '', isActive: false });
    }
  }, [chartOptionFromProp]);

  const baseOptionAfterTypeTransform = useMemo(() => {
    if (!initialLLMOption || !selectedChartType) return null;

    let tempOption = deepClone(initialLLMOption);
    const originalSeriesArray = tempOption.series ? (Array.isArray(tempOption.series) ? tempOption.series : [tempOption.series]) : [];
    const originalXAxis = tempOption.xAxis?.[0] || tempOption.xAxis;
    const originalXAxisData = originalXAxis?.data;

    tempOption.series = originalSeriesArray.map((s: any) => ({
        ...s,
        type: selectedChartType === 'histogram' ? 'bar' : selectedChartType,
        ...(selectedChartType === 'area' && { stack: s.stack || 'total', areaStyle: s.areaStyle || {} }),
        ...(selectedChartType === 'line' && { smooth: typeof s.smooth === 'boolean' ? s.smooth : true }),
        ...(selectedChartType === 'histogram' && { barGap: '0%', barCategoryGap: '20%' }),
    }));

    if (selectedChartType === 'pie' || selectedChartType === 'funnel') {
        tempOption.tooltip = { ...tempOption.tooltip, trigger: 'item' };
        if (llmChartType && llmChartType !== 'pie' && llmChartType !== 'funnel' &&
            originalSeriesArray.length > 0 && originalSeriesArray[0].data &&
            originalXAxisData && Array.isArray(originalXAxisData)) {

            const transformedSeries = originalSeriesArray.map((origSeries: { data: any[]; name: any; }, seriesIdx: number) => {
                if (!origSeries.data || !Array.isArray(origSeries.data)) return null;
                const pieData = origSeries.data.map((val: any, i: number) => ({
                    name: String(originalXAxisData[i] || `Item ${i+1}`),
                    value: typeof val === 'object' && val !== null && val.value !== undefined ? val.value : val,
                }));

                if (pieData.length === 0) return null;

                const numPies = originalSeriesArray.length;
                const pieRadius = numPies > 1 ? `${Math.max(15, 40 - numPies * 5)}%` : '60%';
                const pieCenter = numPies > 1 ? [`${100 / (numPies + 1) * (seriesIdx + 1)}%`, '50%'] : ['50%', '50%'];
                const funnelWidth = numPies > 1 ? `${Math.max(20, 80 / numPies - 5)}%` : '80%';
                const funnelLeft = numPies > 1 ? `${(100 / numPies) * seriesIdx + (100/numPies - parseFloat(funnelWidth))/2}%` : '10%';

                return {
                    name: origSeries.name || `Series ${seriesIdx + 1}`,
                    type: selectedChartType,
                    data: pieData,
                    ...(selectedChartType === 'pie' && { radius: pieRadius, center: pieCenter, label: { show: numPies > 2 ? false : true } }),
                    ...(selectedChartType === 'funnel' && { width: funnelWidth, left: funnelLeft, sort: 'descending', gap: 2 }),
                };
            }).filter((s: null) => s !== null);
            tempOption.series = transformedSeries.length > 0 ? transformedSeries : tempOption.series;
        }
        tempOption.xAxis = undefined; tempOption.yAxis = undefined; tempOption.grid = undefined; tempOption.radar = undefined;

    } else if (selectedChartType === 'radar') {
        tempOption.tooltip = { ...tempOption.tooltip, trigger: 'item' };
        tempOption.xAxis = undefined; tempOption.yAxis = undefined; tempOption.grid = undefined;
        let indicatorData: { name: string, max?: number }[] = [];
        if (originalXAxisData && Array.isArray(originalXAxisData)) {
            indicatorData = originalXAxisData.map((name:any) => ({ name: String(name) }));
        } else if (originalSeriesArray.length > 0 && originalSeriesArray[0].data && (llmChartType === 'pie' || llmChartType === 'funnel') && originalSeriesArray[0].data.every((d:any)=>d.name !== undefined)) {
            indicatorData = originalSeriesArray[0].data.map((d:any) => ({name: String(d.name)}));
        }

        if(indicatorData.length > 0) {
            originalSeriesArray.forEach((series: any) => {
                if(series.data && Array.isArray(series.data)) {
                    series.data.forEach((val:any, idx:number) => {
                        if(indicatorData[idx]) {
                            const numVal = typeof val === 'object' && val !== null && val.value !== undefined ? val.value : val;
                            if(typeof numVal === 'number') {
                                indicatorData[idx].max = Math.max(indicatorData[idx].max || 0, numVal * 1.1);
                            }
                        }
                    });
                }
            });
        }
        tempOption.radar = { ...(initialLLMOption.radar || {}), indicator: indicatorData.length > 0 ? indicatorData : undefined };
        tempOption.series = originalSeriesArray.map((origSeries: any) => ({
            name: origSeries.name, type: 'radar',
            data: [{ value: (llmChartType === 'pie' || llmChartType === 'funnel') ? origSeries.data.map((d:any)=> d.value) : origSeries.data, name: origSeries.name }],
        }));

    } else { // Cartesian charts
        tempOption.tooltip = { ...tempOption.tooltip, trigger: selectedChartType === 'scatter' ? 'item' : 'axis' };
        if (llmChartType === 'pie' || llmChartType === 'funnel' || llmChartType === 'radar' || !initialLLMOption.xAxis) {
            tempOption.xAxis = deepClone(initialLLMOption.xAxis) || { type: 'category', data: [] };
            if ((llmChartType === 'pie' || llmChartType === 'funnel') && originalSeriesArray.length > 0 && originalSeriesArray[0].data?.every((d:any) => d.name !== undefined)) {
                (tempOption.xAxis as any).data = originalSeriesArray[0].data.map((d:any) => d.name);
            }
        } else { tempOption.xAxis = deepClone(initialLLMOption.xAxis); }

        if (llmChartType === 'pie' || llmChartType === 'funnel' || llmChartType === 'radar' || !initialLLMOption.yAxis) {
            tempOption.yAxis = deepClone(initialLLMOption.yAxis) || { type: 'value' };
        } else { tempOption.yAxis = deepClone(initialLLMOption.yAxis); }

        if (llmChartType === 'pie' || llmChartType === 'funnel' || llmChartType === 'radar' || !initialLLMOption.grid) {
            tempOption.grid = deepClone(initialLLMOption.grid) || { containLabel: true, left: '10%', right: '10%', bottom: '15%', top: '15%' };
        } else { tempOption.grid = deepClone(initialLLMOption.grid); }
        tempOption.radar = undefined;

        if (selectedChartType === 'scatter') {
            if(tempOption.xAxis) { Array.isArray(tempOption.xAxis) ? tempOption.xAxis.forEach((ax:any) => ax.type = 'value') : (tempOption.xAxis as any).type = 'value'; }
            if(tempOption.yAxis) { Array.isArray(tempOption.yAxis) ? tempOption.yAxis.forEach((ax:any) => ax.type = 'value') : (tempOption.yAxis as any).type = 'value'; }
        } else {
            if(tempOption.xAxis && !['value', 'time', 'log'].includes( Array.isArray(tempOption.xAxis) ? (tempOption.xAxis[0] as any)?.type : (tempOption.xAxis as any)?.type) ) {
                if(Array.isArray(tempOption.xAxis)) tempOption.xAxis.forEach((ax:any) => ax.type = 'category'); else (tempOption.xAxis as any).type = 'category';
            }
        }
    }
    return tempOption;
  }, [initialLLMOption, selectedChartType, llmChartType]);

  useEffect(() => {
    if (baseOptionAfterTypeTransform) {
      let categories: string[] = [];
      const seriesArray = baseOptionAfterTypeTransform.series ? (Array.isArray(baseOptionAfterTypeTransform.series) ? baseOptionAfterTypeTransform.series : [baseOptionAfterTypeTransform.series]) : [];
      const currentType = seriesArray[0]?.type;

      if (currentType === 'pie' || currentType === 'funnel') {
        if (seriesArray.length > 0 && Array.isArray(seriesArray[0].data)) {
          categories = seriesArray[0].data.map((d: any) => String(d.name)).filter((name:string) => name && name !== 'undefined');
        }
      } else if (baseOptionAfterTypeTransform.xAxis) {
        const xAxis = Array.isArray(baseOptionAfterTypeTransform.xAxis) ? baseOptionAfterTypeTransform.xAxis[0] : baseOptionAfterTypeTransform.xAxis;
        if (Array.isArray(xAxis?.data)) {
          categories = xAxis.data.map(String);
        }
      }
      const uniqueCategories = [...new Set(categories)];
      const prevAvailableCategories = availableFilterCategories;
      setAvailableFilterCategories(uniqueCategories);

      if (selectedChartType !== llmChartType || uniqueCategories.join(',') !== prevAvailableCategories.join(',')) {
        setSelectedFilterCategories(new Set(uniqueCategories));
        setNumericalFilter(prev => ({ ...prev, isActive: false, condition: 'none', value: '' }));
      }

      if (seriesArray.length > 0 && Array.isArray(seriesArray[0].data) && seriesArray[0].data.length > 0) {
        const firstDataPoint = seriesArray[0].data[0];
        if (typeof firstDataPoint === 'number' || (typeof firstDataPoint === 'object' && firstDataPoint !== null && typeof firstDataPoint.value === 'number') || (Array.isArray(firstDataPoint) && typeof firstDataPoint[1] === 'number')) {
          setIsNumericalFilterPossible(true);
        } else { setIsNumericalFilterPossible(false); }
      } else { setIsNumericalFilterPossible(false); }
    } else {
      setAvailableFilterCategories([]); setSelectedFilterCategories(new Set()); setIsNumericalFilterPossible(false);
    }
  }, [baseOptionAfterTypeTransform, selectedChartType, llmChartType]);

  const handleChartTypeChange = (event: SelectChangeEvent<string>) => {
      setSelectedChartType(event.target.value as SwitchableChartType);
      setTransformationError(null);
      setSortConfig({ key: 'value', order: 'none' });
      setNumericalFilter({ condition: 'none', value: '', isActive: false });
  };

  const handleResetToOriginal = useCallback(() => {
      setSelectedChartType(llmChartType);
      setTransformationError(null);
      setSortConfig({ key: 'value', order: 'none' });
      setNumericalFilter({ condition: 'none', value: '', isActive: false });
      setSelectedFilterCategories(new Set(availableFilterCategories));
  }, [llmChartType, availableFilterCategories]);

  const handleSortChange = (event: React.MouseEvent<HTMLElement>, newSortOrder: 'asc' | 'desc' | 'none' | null) => {
    if (newSortOrder !== null) {
        setSortConfig(prev => ({ ...prev, order: newSortOrder as 'asc' | 'desc' | 'none'}));
    }
  };

  const handleFilterCategoryChange = (event: React.SyntheticEvent, newValue: string[]) => {
    setSelectedFilterCategories(new Set(newValue));
  };

  const handleNumericalFilterChange = (key: keyof NumericalFilterConfig, value: any) => {
    setNumericalFilter(prev => {
        const newConfig = {...prev, [key]: value};
        if(key === 'condition') {
            newConfig.isActive = value !== 'none' && newConfig.value !== '';
            if(value === 'none') newConfig.value = '';
        } else if (key === 'value') {
            newConfig.isActive = prev.condition !== 'none' && value !== '';
        }
        return newConfig;
    });
  };

  const optionForECharts = useMemo(() => {
    if (!initialLLMOption) return null;
    if (!selectedChartType || !baseOptionAfterTypeTransform) {
        return initialLLMOption;
    }

    try {
      setTransformationError(null);
      let workingOption = deepClone(baseOptionAfterTypeTransform);

      const currentSeriesArray = workingOption.series ? (Array.isArray(workingOption.series) ? workingOption.series : [workingOption.series]) : [];
      let currentMainSeries = currentSeriesArray.length > 0 ? currentSeriesArray[0] : null;
      let currentXAxis = workingOption.xAxis ? (Array.isArray(workingOption.xAxis) ? workingOption.xAxis[0] : workingOption.xAxis) : null;
      const currentSeriesType = currentMainSeries?.type;

      // Apply Categorical Filtering
      if (availableFilterCategories.length > 0 && selectedFilterCategories.size !== availableFilterCategories.length) {
        if (currentSeriesType === 'pie' || currentSeriesType === 'funnel') {
          if (currentMainSeries && Array.isArray(currentMainSeries.data)) {
            currentMainSeries.data = currentMainSeries.data.filter((item: any) => selectedFilterCategories.has(String(item.name)));
          }
        } else if (currentXAxis && Array.isArray(currentXAxis.data)) {
          const indicesToKeep: number[] = [];
          const filteredXCategories: string[] = [];
          currentXAxis.data.forEach((cat: any, index: number) => {
            if (selectedFilterCategories.has(String(cat))) {
              indicesToKeep.push(index);
              filteredXCategories.push(String(cat));
            }
          });
          currentXAxis.data = filteredXCategories;
          currentSeriesArray.forEach((s: any) => {
            if (Array.isArray(s.data)) {
              s.data = indicesToKeep.map(index => s.data[index]).filter(x => x !== undefined);
            }
          });
        }
      }
      if (selectedFilterCategories.size === 0 && availableFilterCategories.length > 0) {
        currentSeriesArray.forEach((s:any) => { if(s) s.data = []; });
        if(currentXAxis) currentXAxis.data = [];
      }

      // Apply Numerical Filtering
      if (numericalFilter.isActive && numericalFilter.condition !== 'none' && numericalFilter.value !== '') {
        const filterVal = parseFloat(numericalFilter.value);
        if (!isNaN(filterVal) && currentMainSeries && Array.isArray(currentMainSeries.data)) {
          const dataForNumFilter = [...currentMainSeries.data];
          const categoriesForNumFilter = currentXAxis?.data ? [...currentXAxis.data] : [];
          const indicesToKeepNum: number[] = [];

          dataForNumFilter.forEach((dataPoint: any, index: number) => {
            let pointValue: number | undefined;
            if (typeof dataPoint === 'number') pointValue = dataPoint;
            else if (typeof dataPoint === 'object' && dataPoint !== null && typeof dataPoint.value === 'number') pointValue = dataPoint.value;
            else if (Array.isArray(dataPoint) && typeof dataPoint[1] === 'number') pointValue = dataPoint[1];

            if (pointValue !== undefined) {
              let satisfies = false;
              switch (numericalFilter.condition) {
                case '>': { satisfies = pointValue > filterVal; break; }
                case '>=': { satisfies = pointValue >= filterVal; break; }
                case '<': { satisfies = pointValue < filterVal; break; }
                case '<=': { satisfies = pointValue <= filterVal; break; }
                case '==': { satisfies = pointValue === filterVal; break; }
              }
              if (satisfies) indicesToKeepNum.push(index);
            }
          });

          if (currentSeriesType !== 'pie' && currentSeriesType !== 'funnel' && currentXAxis && Array.isArray(currentXAxis.data)) {
             currentXAxis.data = indicesToKeepNum.map(index => categoriesForNumFilter[index]).filter(x => x !== undefined);
          }
          currentSeriesArray.forEach((s: any) => {
            if (Array.isArray(s.data)) {
              s.data = indicesToKeepNum.map(index => s.data[index]).filter(x => x !== undefined);
            }
          });
        }
      }
      workingOption.series = currentSeriesArray;
      if(currentXAxis) { if(Array.isArray(workingOption.xAxis)) workingOption.xAxis[0] = currentXAxis; else workingOption.xAxis = currentXAxis; }

      // Apply Sorting
      const seriesToSort = workingOption.series?.[0];
      const xAxisForSorting = workingOption.xAxis ? (Array.isArray(workingOption.xAxis) ? workingOption.xAxis[0] : workingOption.xAxis) : null;

      if (sortConfig.order !== 'none' && seriesToSort && Array.isArray(seriesToSort.data) && seriesToSort.data.length > 0) {
          if (currentSeriesType === 'pie' || currentSeriesType === 'funnel') {
            seriesToSort.data.sort((a: any, b: any) => {
                const valA = sortConfig.key === 'name' ? String(a.name || '') : Number(a.value || 0);
                const valB = sortConfig.key === 'name' ? String(b.name || '') : Number(b.value || 0);
                if (valA < valB) return sortConfig.order === 'asc' ? -1 : 1;
                if (valA > valB) return sortConfig.order === 'asc' ? 1 : -1;
                return 0;
            });
          } else if (xAxisForSorting && Array.isArray(xAxisForSorting.data) && seriesToSort.data.length === xAxisForSorting.data.length) {
              const combined = xAxisForSorting.data.map((cat: any, index: number) => ({
                category: String(cat),
                value: (typeof seriesToSort.data[index] === 'object' && seriesToSort.data[index] !== null && seriesToSort.data[index].value !== undefined) ? seriesToSort.data[index].value : seriesToSort.data[index],
                originalIndexInData: index,
              }));
              combined.sort((a: any, b: any) => {
                const valA = sortConfig.key === 'name' ? a.category : Number(a.value || 0);
                const valB = sortConfig.key === 'name' ? b.category : Number(b.value || 0);
                if (valA < valB) return sortConfig.order === 'asc' ? -1 : 1;
                if (valA > valB) return sortConfig.order === 'asc' ? 1 : -1;
                return 0;
              });
              xAxisForSorting.data = combined.map((item: { category: any; }) => item.category);
workingOption.series.forEach((s: any) => {
    if (s.data && Array.isArray(s.data)) {
      const reorderedData = new Array(s.data.length);
      
      // The types for 'sortedItem' and 'newIndex' are now correct.
      combined.forEach((sortedItem: { originalIndexInData: number }, newIndex: number) => {
        reorderedData[newIndex] = s.data[sortedItem.originalIndexInData];
      });

      s.data = reorderedData.filter((x: any) => x !== undefined);
    }
});
          }
      }

      // Final Merge with Global Defaults
      const globalDefaults = {
        title: { left: 'center', top: 10, textStyle: { fontSize: 16, fontWeight: 'bold' }},
        legend: { show: true, orient: 'horizontal', left: 'center', bottom: 5, type: 'scroll', textStyle: { fontSize: 12 }},
        toolbox: { show: true, orient: 'vertical', left: 'right', top: 'middle', feature: { mark: { show: false }, dataView: { show: true, readOnly: true, title: 'View Data' }, restore: { show: true, title: 'Restore' }, saveAsImage: { show: true, title: 'Save as Image', pixelRatio: 2 }}},
        responsive: true,
        tooltip: {
            trigger: 'axis',
            confine: true, // Prevents tooltip from going off-screen
        },
      };
      const finalMergedOption = {
          ...globalDefaults, ...workingOption,
          title: { ...globalDefaults.title, ...(workingOption.title || {}) },
          tooltip: { ...globalDefaults.tooltip, ...workingOption.tooltip },
          legend: { ...globalDefaults.legend, ...(workingOption.legend || {}) },
          grid: workingOption.grid, xAxis: workingOption.xAxis, yAxis: workingOption.yAxis, radar: workingOption.radar,
          toolbox: { ...globalDefaults.toolbox, ...(workingOption.toolbox || {}), feature: { ...globalDefaults.toolbox.feature, ...(workingOption.toolbox?.feature || {}), magicType: { show: false }}},
      };

      // Final Axis Fix
      const finalSeriesType = finalMergedOption.series?.[0]?.type;
      if (['bar', 'line', 'area', 'histogram'].includes(finalSeriesType || '')) {
        let xAxisIsEffectivelyPresent = finalMergedOption.xAxis && (Array.isArray(finalMergedOption.xAxis) ? finalMergedOption.xAxis.length > 0 && finalMergedOption.xAxis[0] !== undefined : true);
        let yAxisIsEffectivelyPresent = finalMergedOption.yAxis && (Array.isArray(finalMergedOption.yAxis) ? finalMergedOption.yAxis.length > 0 && finalMergedOption.yAxis[0] !== undefined : true);
        if (xAxisIsEffectivelyPresent && !yAxisIsEffectivelyPresent) {
            finalMergedOption.yAxis = { type: 'value', ...((Array.isArray(finalMergedOption.yAxis) ? finalMergedOption.yAxis[0] : finalMergedOption.yAxis) || {}) };
        } else if (yAxisIsEffectivelyPresent && !xAxisIsEffectivelyPresent) {
            finalMergedOption.xAxis = { type: 'category', ...((Array.isArray(finalMergedOption.xAxis) ? finalMergedOption.xAxis[0] : finalMergedOption.xAxis) || {}) };
        } else if (!xAxisIsEffectivelyPresent && !yAxisIsEffectivelyPresent && finalMergedOption.series?.[0]?.data?.length > 0) {
             finalMergedOption.xAxis = { type: 'category', data: finalMergedOption.series[0].data.map((_:any, i:number) => `Category ${i+1}`) };
             finalMergedOption.yAxis = { type: 'value' };
             finalMergedOption.grid = finalMergedOption.grid || { containLabel: true, left: '3%', right: '4%', bottom: '12%', top: '15%' };
        }
      }
      return finalMergedOption;

    } catch (error: any) {
      console.error("Error during chart option computation:", error);
      setTransformationError(`Failed to apply transformations: ${error.message}. Displaying original chart.`);
      return initialLLMOption;
    }
  }, [initialLLMOption, selectedChartType, llmChartType, sortConfig, selectedFilterCategories, availableFilterCategories, baseOptionAfterTypeTransform, numericalFilter]);

  if (!initialLLMOption) return <Box sx={{ p: 3, textAlign: 'center', height: '100%' }}><Typography>No chart data loaded.</Typography></Box>;
  const hasSeries = initialLLMOption.series && ((Array.isArray(initialLLMOption.series) && initialLLMOption.series.length > 0) || typeof initialLLMOption.series === 'object');
  const hasDataset = initialLLMOption.dataset?.source?.length > 0;
  if (!hasSeries && !hasDataset && (!optionForECharts || !optionForECharts.series?.length)) {
      return <Box sx={{ p: 3, textAlign: 'center', height: '100%' }}><Typography>Chart data is incomplete.</Typography></Box>;
  }
  if (!optionForECharts) return <Box sx={{ p: 3, textAlign: 'center', height: '100%' }}><Typography>Preparing chart...</Typography></Box>;

  return (
    <Box sx={{ width: '100%', height: '100%', p: 1, display: 'flex', flexDirection: 'column' }}>
      <Box sx={{display: 'flex', justifyContent: 'flex-start', mb: controlsOpen ? 1 : 0 }}>
        <Button
            variant="text" size="small" color="primary"
            onClick={() => setControlsOpen(prev => !prev)}
            startIcon={controlsOpen ? <CloseIcon /> : <TuneIcon />}
            sx={{ textTransform: 'none', alignSelf: 'flex-start' }}
        >
          {controlsOpen ? 'Hide Controls' : 'Chart Controls'}
        </Button>
      </Box>

      <Collapse in={controlsOpen} timeout="auto" unmountOnExit>
        <Paper elevation={0} variant="outlined" sx={{ p: 2, mb: 2, borderRadius: 2 }}>
          <Stack spacing={2.5}>
            <Stack direction={{xs: "column", sm: "row"}} spacing={2} alignItems="center" flexWrap="wrap">
              <FormControl sx={{ minWidth: 180, flexBasis: 180, flexGrow: { xs: 1, sm: 0 } }} size="small" variant="outlined">
                <InputLabel id="chart-type-select-label">Chart Type</InputLabel>
                <Select
                  labelId="chart-type-select-label" value={selectedChartType || ''} label="Chart Type"
                  onChange={handleChartTypeChange}
                  disabled={compatibleChartTypes.length === 0 && !selectedChartType}
                >
                  {SUPPORTED_CHART_TYPES_FOR_SWITCHING.map((type) => (
                    <MenuItem key={type} value={type} disabled={!compatibleChartTypes.includes(type)}>
                      {type.charAt(0).toUpperCase() + type.slice(1)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Box>
                <Typography variant="caption" display="block" sx={{mb:0.5, color: 'text.secondary', textAlign:'left'}}>Sort by Value</Typography>
                <ToggleButtonGroup value={sortConfig.order} exclusive onChange={handleSortChange} size="small">
                    <ToggleButton value="none" title="Default Order"><SortByAlphaIcon /></ToggleButton>
                    <ToggleButton value="asc" title="Sort Ascending"><ArrowUpwardIcon /></ToggleButton>
                    <ToggleButton value="desc" title="Sort Descending"><ArrowDownwardIcon /></ToggleButton>
                </ToggleButtonGroup>
              </Box>
              <Box sx={{flexGrow: 1, display: {xs: 'none', md: 'block'}}} />
              <Button variant="outlined" color="inherit" size="small" onClick={handleResetToOriginal} disabled={selectedChartType === llmChartType && sortConfig.order === 'none' && selectedFilterCategories.size === availableFilterCategories.length && !numericalFilter.isActive} startIcon={<RestartAltIcon/>}>
                  Reset All
              </Button>
            </Stack>

            <Stack direction={{xs:"column", lg:"row"}} spacing={2} alignItems={{xs:"stretch", lg:"flex-start"}}>
                {availableFilterCategories.length > 0 && (
                <FormControl sx={{ minWidth: 200, flexBasis: 300, flexGrow: 1 }} variant="outlined" >
                    <Autocomplete
                        multiple size="small" options={availableFilterCategories}
                        value={Array.from(selectedFilterCategories)}
                        onChange={handleFilterCategoryChange}
                        disableCloseOnSelect
                        getOptionLabel={(option) => option}
                        renderTags={(value: readonly string[], getTagProps) => {
                            const numSelected = value.length;
                            if (numSelected === 0) return null;
                            const label = numSelected === availableFilterCategories.length && availableFilterCategories.length > 0
                                ? 'All Categories'
                                : `<span class="math-inline">\{numSelected\} Categor</span>{numSelected === 1 ? 'y' : 'ies'} Selected`;
                            return [<Chip key="cat-summary-chip" size="small" label={label} sx={{maxWidth: 'calc(100% - 30px)'}} />];
                        }}
                        renderOption={(props, option, { selected }) => (
                            <li {...props}>
                            <Checkbox icon={<CheckBoxOutlineBlankIcon fontSize="small" />} checkedIcon={<CheckBoxIcon fontSize="small" />} sx={{ mr: 1, p:0}} checked={selected}/>
                            <Typography variant="body2" sx={{fontSize: '0.875rem'}}>{option}</Typography>
                            </li>
                        )}
                        renderInput={(params) => (
                            <TextField {...params} variant="outlined" label="Filter Categories" placeholder={selectedFilterCategories.size > 0 ? "" : (availableFilterCategories.length > 3 ? "Categories..." : "")} />
                        )}
                        ListboxProps={{ style: { maxHeight: 180, overflow: 'auto' } }}
                    />
                </FormControl>
                )}

                {isNumericalFilterPossible && (
                <Box sx={{ minWidth: 240, flexBasis: 320, flexGrow: 1 }}>
                    <Typography variant="caption" display="block" sx={{mb:0.5, color: 'text.secondary'}}>
                        <FunctionsIcon fontSize="inherit" sx={{verticalAlign: 'middle', mr:0.5}}/>Filter by Value (First Series)
                    </Typography>
                    <Stack direction="row" spacing={1} alignItems="center">
                        <FormControl size="small" sx={{minWidth: 100, flexBasis: 130}} variant="outlined">
                        <InputLabel>Condition</InputLabel>
                        <Select
                            value={numericalFilter.condition} label="Condition"
                            onChange={(e) => handleNumericalFilterChange('condition', e.target.value)}
                        >
                            <MenuItem value="none"><em>None</em></MenuItem>
                            <MenuItem value=">">&gt;</MenuItem>
                            <MenuItem value=">=">&ge;</MenuItem>
                            <MenuItem value="<">&lt;</MenuItem>
                            <MenuItem value="<=">&le;</MenuItem>
                            <MenuItem value="==">=</MenuItem>
                        </Select>
                        </FormControl>
                        <TextField
                            type="number" size="small" label="Value" variant="outlined"
                            value={numericalFilter.value}
                            onChange={(e) => handleNumericalFilterChange('value', e.target.value)}
                            disabled={numericalFilter.condition === 'none'}
                            InputLabelProps={{ shrink: true }}
                            sx={{flexGrow: 1, flexBasis: 100}}
                        />
                    </Stack>
                </Box>
                )}
            </Stack>
          </Stack>
        </Paper>
      </Collapse>

      {transformationError && ( <Alert severity="warning" sx={{ mt:1, mb: 1 }}> {transformationError} </Alert> )}

      <Box sx={{ flexGrow: 1, width: '100%', minHeight: 0 }}>
        <ReactECharts
            option={optionForECharts}
            notMerge={true}
            lazyUpdate={false}
          style={{ height: "100%", width: "100%", minHeight: "350px" }}
          onChartReady={(instance) => onChartRendered(instance)}
        />
      </Box>
    </Box>
  );
}