#property script_show_inputs
input string InTemplateName = "Auto_MA.tpl";
input string InIndicatorPath = "Examples\\Moving Average";

void OnStart()
{
   long chart = ChartID();
   string sym = Symbol();
   ENUM_TIMEFRAMES tf = (ENUM_TIMEFRAMES)Period();
   PrintFormat("[TemplateBuilder] chart=%I64d %s %s", chart, sym, EnumToString(tf));
   int handle = iCustom(sym, tf, InIndicatorPath);
   if(handle == INVALID_HANDLE)
   {
      PrintFormat("[TemplateBuilder] iCustom falhou (%s): %d", InIndicatorPath, GetLastError());
      return;
   }
   if(!ChartIndicatorAdd(chart, 0, handle))
   {
      PrintFormat("[TemplateBuilder] ChartIndicatorAdd falhou: %d", GetLastError());
   }
   ChartRedraw();
   Sleep(1000);
   if(ChartSaveTemplate(chart, InTemplateName))
      PrintFormat("[TemplateBuilder] Template salvo: %s", InTemplateName);
   else
      PrintFormat("[TemplateBuilder] ChartSaveTemplate falhou: %d", GetLastError());
}
