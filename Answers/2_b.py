import pandas

def RowWithColumn (
   rowVar,          # Row variable
   columnVar,       # Column predictor
   show = 'ROW'):   # Show ROW fraction, COLUMN fraction, or BOTH table

   countTable = pandas.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
   print("Frequency Table: \n", countTable)
   print( )

   if (show == 'ROW' or show == 'BOTH'):
       rowFraction = countTable.div(countTable.sum(1), axis='index')
       print("Row Fraction Table: \n", rowFraction)
       print( )

   if (show == 'COLUMN' or show == 'BOTH'):
       columnFraction = countTable.div(countTable.sum(0), axis='columns')
       print("Column Fraction Table: \n", columnFraction)
       print( )

   return

inputData = pandas.read_excel('claim_history.xlsx')

# EBilling -> CreditCard, Gender, JobCategory
subData = inputData[['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']].dropna()

RowWithColumn(rowVar = subData['CAR_USE'], columnVar = subData['CAR_TYPE'], show = 'ROW')
RowWithColumn(rowVar = subData['CAR_USE'], columnVar = subData['OCCUPATION'], show = 'ROW')
RowWithColumn(rowVar = subData['CAR_USE'], columnVar = subData['EDUCATION'], show = 'ROW')