# Logistic Regression


During this class we will:

- Interpret logistic regression coefficients
- Review assumption diagnostics and remedies
- Expand diagnostics and remedies to GLMs

Start by downloading the leads data. To share their solution for
Exercise 13, the randomly selected student is \*\*\_\_\_\*\*.

## Learn

You can also download the slides as an .html file. Once you’ve previewed
the material and identified any questions, start watching the lecture.

We want to keep our model as simple as possible, so why would we use
multiple logistic regression? Go the discussions to share before
continuing with the lecture.

What data are you considering for your second project? Discuss validity
and representativeness as a group. Go the discussions to share before
finishing the lecture.

## Data Dictionary

Sales leads data for B2B electronic sales. Each of the rows represents
one lead along with its resulting Stage (i.e., Qualified or
Disqualified) along with various lead features.

- **Stage**: Progress through the sales development pipeline
- **Industry**: The type of product or service the business provides
- **Employees**: The size of company based on the number of employees
- **TimeZone**: The Lead’s time zone
- **LeadSource**: The source of the data for the Lead
- **days_elapsed**: Number of days elapsed since Lead creation
- **created_quarter**: Quarter of the Lead creation
- **contact_quarter**: Quarter when the Lead was first contacted
- **latest_quarter**: Quarter when the Lead was last contacted
- **EmployeeId**: Unique ID for the sales rep assigned to the Lead
- **ActivityTypeEmail**: Count of email contacts with the Lead
- **ActivityTypePhone Call**: Count of call contacts with the Lead
- **ActivityTypeEmail Response**: Count of email responses with the Lead
- **ActivityTypeMeeting**: Count of meetings with the Lead
- **ActivityTypeLead Handraise**: Count of times Lead requesting
  information
- **ActivityTypeWeb Schedule**: Count of times Lead scheduling an
  appointment
- **Amount**: Estimate of how much the Lead is worth if closed

## Apply

### Exercise 14

1.  Clean up the leads data
2.  Create two interesting visualizations that help you understand the
    data and its limitations
3.  Walk through the logistic regression model assumptions, use
    diagnostics, and justify whether or not they are satisfied for your
    cleaned data
4.  Submit your code, output, and explanations as a single PDF on Canvas

### Milestone 14

Identify and acquire a dataset for your group project. Work through and
validate all of the model assumptions for logistic regression.
