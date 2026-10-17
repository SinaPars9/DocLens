
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


OUTPUT_FILE = "northstar_annual_report_2025.pdf"


def add_page_number(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)

    canvas.drawString(
        20 * mm,
        12 * mm,
        "Northstar Office Supplies | Annual Review 2025",
    )

    canvas.drawRightString(
        190 * mm,
        12 * mm,
        f"Page {document.page}",
    )

    canvas.restoreState()


pdf = SimpleDocTemplate(
    OUTPUT_FILE,
    pagesize=A4,
    rightMargin=20 * mm,
    leftMargin=20 * mm,
    topMargin=20 * mm,
    bottomMargin=22 * mm,
)

styles = getSampleStyleSheet()

styles.add(
    ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=13,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.darkslategray,
        spaceAfter=18,
    )
)

styles.add(
    ParagraphStyle(
        name="ReportBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        spaceAfter=10,
    )
)


def add_paragraph(story, text, style="ReportBody"):
    story.append(Paragraph(text, styles[style]))


def add_heading(story, text):
    story.append(Spacer(1, 10))
    story.append(Paragraph(text, styles["Heading1"]))
    story.append(Spacer(1, 6))


def add_table(story, data, col_widths):
    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.append(table)
    story.append(Spacer(1, 12))


content = []


# --------------------------------------------------
# Cover page
# --------------------------------------------------

content.append(Spacer(1, 65 * mm))

add_paragraph(
    content,
    "Northstar Office Supplies",
    "Title",
)

add_paragraph(
    content,
    "Annual Business Review 2025",
    "ReportSubtitle",
)

add_paragraph(
    content,
    "A fictional business report covering financial performance, "
    "sales channels, customer activity, and operational priorities. "
    "All company names and figures are fictional and intended "
    "for educational testing only.",
)

content.append(PageBreak())


# --------------------------------------------------
# 1. Company overview
# --------------------------------------------------

add_heading(content, "1. Company Overview")

add_paragraph(
    content,
    "Northstar Office Supplies is a fictional office-products retailer "
    "based in Manchester, United Kingdom. Established in 2018, the "
    "company serves small businesses, home offices, schools, and "
    "independent professional customers. Its catalogue includes paper "
    "products, printer supplies, writing instruments, and desk accessories."
)

add_paragraph(
    content,
    "The company sells through its website and through orders placed "
    "by email and telephone. Management considers reliable delivery, "
    "consistent product availability, and responsive customer service "
    "to be important drivers of repeat business. At the end of 2025, "
    "Northstar employed 24 people across customer service, sales, "
    "warehouse operations, purchasing, and administration."
)

add_paragraph(
    content,
    "During 2025, management focused on improving order fulfilment "
    "and making the purchasing process easier for business customers. "
    "The company reviewed its most frequently ordered products, "
    "adjusted stock levels for selected items, and introduced "
    "additional checks before parcels left the warehouse."
)

add_paragraph(
    content,
    "These changes were intended to reduce dispatch delays and "
    "prevent incorrect items from being sent to customers. Although "
    "the business continued to experience occasional supplier delays, "
    "management reported improvements in several operational measures "
    "during the year."
)


# --------------------------------------------------
# 2. Financial performance
# --------------------------------------------------

add_heading(content, "2. Financial Performance")

add_paragraph(
    content,
    "Revenue increased from £2,100,000 in 2024 to £2,400,000 in 2025. "
    "This represents an increase of £300,000, or approximately 14.3%. "
    "The company attributed the improvement to stronger online sales, "
    "repeat orders from existing customers, and improved availability "
    "of several popular product categories."
)

financial_data = [
    ["Financial metric", "2023", "2024", "2025"],
    ["Revenue", "£1,850,000", "£2,100,000", "£2,400,000"],
    ["Cost of goods sold", "£1,110,000", "£1,260,000", "£1,416,000"],
    ["Gross profit", "£740,000", "£840,000", "£984,000"],
    ["Operating expenses", "£590,000", "£630,000", "£708,000"],
    ["Operating profit", "£150,000", "£210,000", "£276,000"],
]

add_table(
    content,
    financial_data,
    [155, 85, 85, 85],
)

add_paragraph(
    content,
    "Gross profit increased to £984,000 in 2025. It is calculated "
    "by subtracting the cost of goods sold from revenue. The cost of "
    "goods sold includes the purchase cost of products sold to customers "
    "during the reporting period. Gross profit does not include all "
    "the costs of running the business."
)

add_paragraph(
    content,
    "Operating expenses reached £708,000 in 2025 and included staff "
    "costs, warehouse expenses, software subscriptions, marketing, "
    "and administration. After these operating expenses were deducted "
    "from gross profit, the company recorded an operating profit of "
    "£276,000. The operating profit margin was 11.5% of revenue."
)

add_paragraph(
    content,
    "Operating profit increased by £66,000 compared with 2024, "
    "when it was £210,000. Management considered this improvement "
    "encouraging but noted that higher sales volumes also increased "
    "warehouse workload and distribution costs. Future profitability "
    "will depend on controlling these costs while maintaining "
    "customer service standards."
)


# --------------------------------------------------
# 3. Sales channels
# --------------------------------------------------

add_heading(content, "3. Sales Channel Analysis")

add_paragraph(
    content,
    "Northstar received orders through three primary channels in 2025: "
    "its website, email, and telephone. The website was the largest "
    "channel by revenue, accounting for 55% of total sales. Email "
    "orders contributed 25%, while telephone orders accounted for "
    "the remaining 20%. These channels serve customers with different "
    "purchasing habits and support requirements."
)

sales_data = [
    [
        "Sales channel",
        "Revenue",
        "Share of sales",
        "Average order",
    ],
    ["Website", "£1,320,000", "55%", "£82"],
    ["Email", "£600,000", "25%", "£116"],
    ["Telephone", "£480,000", "20%", "£143"],
    ["Total", "£2,400,000", "100%", "—"],
]

add_table(
    content,
    sales_data,
    [105, 105, 100, 100],
)

add_paragraph(
    content,
    "Website customers generally placed smaller orders and used the "
    "online catalogue to compare products and check availability. "
    "The website was particularly important for frequently purchased "
    "items such as paper, pens, notebooks, and printer supplies. "
    "Management planned to improve product descriptions and make "
    "reordering common items easier during 2026."
)

add_paragraph(
    content,
    "Email orders had a higher average order value than website orders. "
    "This channel was often used by small businesses that requested "
    "quotes, combined several product categories into one order, "
    "or needed assistance confirming product specifications. "
    "Customer service staff reviewed these requests and helped "
    "customers resolve questions before an order was processed."
)

add_paragraph(
    content,
    "Telephone orders recorded the highest average order value "
    "among the three channels. They were commonly associated with "
    "customers who preferred direct assistance or needed to clarify "
    "availability and delivery arrangements. However, telephone "
    "orders required more staff time per transaction than routine "
    "website orders. Management therefore aimed to preserve the "
    "service benefits of telephone support while encouraging customers "
    "to use self-service options when appropriate."
)

add_paragraph(
    content,
    "The company processed 4,800 orders during 2025. Of these, "
    "1,920 orders came from returning customers, representing "
    "40% of all orders. Management views repeat purchasing as an "
    "important indicator of customer retention, although order "
    "frequency alone does not explain why customers choose to return. "
    "Product availability, delivery reliability, price, and the "
    "quality of customer support may all influence this behaviour."
)

add_paragraph(
    content,
    "The figures in this section describe different measures. "
    "Revenue represents the total sales value for a channel, while "
    "average order value represents the average amount spent per order "
    "within that channel. A channel with a smaller share of total "
    "revenue may still have a higher average order value because "
    "its customers place fewer but larger orders."
)


# --------------------------------------------------
# 4. Definitions
# --------------------------------------------------

add_heading(content, "4. Key Financial Definitions")

add_paragraph(
    content,
    "<b>Revenue:</b> The total value of sales generated during a "
    "reporting period before deducting business expenses."
)

add_paragraph(
    content,
    "<b>Cost of goods sold:</b> The direct cost associated with "
    "the products sold during the reporting period."
)

add_paragraph(
    content,
    "<b>Gross profit:</b> Revenue minus the cost of goods sold."
)

add_paragraph(
    content,
    "<b>Operating expenses:</b> The costs of running the business "
    "that are deducted from gross profit when calculating "
    "operating profit."
)

add_paragraph(
    content,
    "<b>Operating profit:</b> Gross profit minus operating expenses, "
    "before interest and tax."
)

add_paragraph(
    content,
    "<b>Average order value:</b> The average revenue generated "
    "per order in a particular sales channel."
)

add_paragraph(
    content,
    "<b>Returning-customer order:</b> An order placed by a customer "
    "who has purchased from the company before. In this report, "
    "the returning-customer percentage is calculated using the "
    "number of orders, not the number of individual customers."
)


# --------------------------------------------------
# 5. Product category performance
# --------------------------------------------------

add_heading(content, "5. Product Category Performance")

add_paragraph(
    content,
    "Northstar organises its catalogue into five product categories. "
    "Tracking these categories helps management see which products are "
    "driving growth and which may need more promotional support. In 2025, "
    "paper products remained the largest category by revenue, while desk "
    "accessories accounted for 10% of total revenue.."
)

category_data = [
    ["Product category", "2023", "2024", "2025", "2025 share"],
    ["Paper products", "£740,000", "£840,000", "£960,000", "40%"],
    ["Printer supplies", "£555,000", "£630,000", "£720,000", "30%"],
    ["Writing instruments", "£277,500", "£315,000", "£360,000", "15%"],
    ["Desk accessories", "£185,000", "£210,000", "£240,000", "10%"],
    ["Other products", "£92,500", "£105,000", "£120,000", "5%"],
    ["Total", "£1,850,000", "£2,100,000", "£2,400,000", "100%"],
]

add_table(content, category_data, [110, 80, 80, 80, 60])

add_paragraph(
    content,
    "Paper products include copier paper, stationery paper, and specialty "
    "media. Revenue in this category grew from £740,000 in 2023 to "
    "£960,000 in 2025, an increase of £220,000. The growth was driven by "
    "higher order volumes from schools and the introduction of a premium "
    "recycled-paper range that carried a higher price point."
)

add_paragraph(
    content,
    "Printer supplies cover ink, toner, drums, and maintenance kits. These "
    "products carry high margins because they are frequently reordered. "
    "Revenue rose from £555,000 to £720,000 between 2023 and 2025. "
    "Management expects continued demand as customers install new printers "
    "and maintain existing hardware throughout the year."
)

add_paragraph(
    content,
    "Writing instruments include pens, pencils, markers, and correction "
    "products. The category grew modestly from £277,500 to £360,000 over "
    "the three-year period. Pens and pencils account for most of the "
    "volume, while branded premium pens contribute disproportionately to "
    "overall revenue."
)


add_paragraph(
    content,
    "Desk accessories is one of the smaller product categories, accounting "
    "for 10% of total revenue in 2025. All product categories recorded "
    "similar revenue growth between 2023 and 2025. Revenue in this category "
    "reached £240,000 in 2025. The business treats this category as a "
    "priority for margin expansion because of its higher average order values."
)



# --------------------------------------------------
# 6. Geographic distribution
# --------------------------------------------------

add_heading(content, "6. Geographic Distribution of Sales")

add_paragraph(
    content,
    "Northstar sells throughout the United Kingdom and organises its "
    "distribution into four regions plus an online-only segment. Each "
    "region is supported by a dedicated territory manager who works with "
    "local schools, offices, and resellers. Regional performance varied "
    "in 2025, with the South region contributing the largest share of "
    "total revenue."
)

region_data = [
    ["Region", "2023", "2024", "2025", "2025 share"],
    ["North", "£560,000", "£640,000", "£720,000", "30%"],
    ["South", "£740,000", "£840,000", "£960,000", "40%"],
    ["East", "£280,000", "£320,000", "£360,000", "15%"],
    ["West", "£180,000", "£210,000", "£240,000", "10%"],
    ["Online only", "£90,000", "£90,000", "£120,000", "5%"],
    ["Total", "£1,850,000", "£2,100,000", "£2,400,000", "100%"],
]

add_table(content, region_data, [110, 80, 80, 80, 60])

add_paragraph(
    content,
    "The South region, which includes London and the surrounding "
    "metropolitan area, delivered £960,000 in revenue in 2025. This "
    "represented 40% of the company's total sales. The region benefits "
    "from a high concentration of small businesses and government offices "
    "that place frequent orders for standard office supplies."
)

add_paragraph(
    content,
    "The North region generated £720,000 in 2025, accounting for 30% of "
    "revenue. Growth over the three years was steady but slower than the "
    "South, partly because the customer base skews toward larger "
    "corporate accounts with longer procurement cycles."
)

add_paragraph(
    content,
    "The East and West regions together contributed 25% of total revenue. "
    "Both regions are served from a single distribution hub located near "
    "Manchester, which helps keep delivery costs low but limits how "
    "quickly very large orders can be fulfilled."
)

add_paragraph(
    content,
    "The online-only segment, which includes orders placed through the "
    "website from customers who do not fall into a named region, grew "
    "to £120,000 in 2025. Management expects this segment to expand "
    "faster than the rest of the business over the next two years."
)


# --------------------------------------------------
# 7. Customer segments
# --------------------------------------------------

add_heading(content, "7. Customer Segments")

add_paragraph(
    content,
    "Northstar classifies its customers into four segments based on order "
    "volume and purchasing behaviour. Understanding these segments helps "
    "the company tailor its pricing, service levels, and marketing. In "
    "2025, small businesses formed the largest segment by both order count "
    "and revenue, while independent professionals placed the fewest but "
    "highest-value orders."
)

segment_data = [
    ["Customer segment", "Orders", "Share", "Revenue", "Avg order"],
    ["Small businesses", "2,400", "50%", "£1,200,000", "£500"],
    ["Home offices", "1,440", "30%", "£576,000", "£400"],
    ["Schools", "720", "15%", "£432,000", "£600"],
    ["Independent professionals", "240", "5%", "£192,000", "£800"],
    ["Total", "4,800", "100%", "£2,400,000", "£500"],
]

add_table(content, segment_data, [150, 65, 55, 80, 70])

add_paragraph(
    content,
    "Small businesses, defined as organisations with fewer than 50 "
    "employees, placed 2,400 orders in 2025. These orders accounted for "
    "£1,200,000 in revenue, exactly half of the total. These customers "
    "typically reorder every few weeks and respond well to bulk-pricing "
    "offers sent by email."
)

add_paragraph(
    content,
    "Home offices, comprising remote workers and sole proprietors, "
    "placed 1,440 orders worth £576,000. Their average order value of "
    "£400 was below the company average because they tend to buy "
    "smaller quantities of everyday items such as paper and pens."
)

add_paragraph(
    content,
    "Schools placed 720 orders in 2025, generating £432,000 in revenue. "
    "Although their order count was lower, their average order value of "
    "£600 was above the company average. Schools typically place larger "
    "orders at the start of each term and prefer to pay on invoice."
)

add_paragraph(
    content,
    "Independent professionals, such as freelancers and consultants, "
    "placed the fewest orders at 240 but achieved the highest average "
    "order value of £800. They often buy specialised or premium products "
    "and value fast, reliable delivery."
)


# --------------------------------------------------
# 8. Quarterly performance
# --------------------------------------------------

add_heading(content, "8. Quarterly Performance 2025")

add_paragraph(
    content,
    "This section breaks down the company's 2025 performance by quarter. "
    "Revenue was highest in the final quarter, reflecting stronger "
    "demand for back-to-school supplies and end-of-year restocking. "
    "Profitability improved through the year as operating expenses grew "
    "more slowly than revenue."
)

quarterly_data = [
    ["Quarter", "Revenue", "Operating profit", "Margin", "Orders"],
    ["Q1 (Jan-Mar)", "£540,000", "£54,000", "10.0%", "1,100"],
    ["Q2 (Apr-Jun)", "£588,000", "£65,000", "11.1%", "1,200"],
    ["Q3 (Jul-Sep)", "£624,000", "£72,000", "11.5%", "1,240"],
    ["Q4 (Oct-Dec)", "£648,000", "£85,000", "13.1%", "1,260"],
    ["Total", "£2,400,000", "£276,000", "11.5%", "4,800"],
]

add_table(content, quarterly_data, [90, 95, 100, 65, 60])

add_paragraph(
    content,
    "Quarterly revenue grew from £540,000 in the first quarter to "
    "£648,000 in the fourth quarter. This represents an increase of "
    "£108,000 between the opening and closing quarters. The steady "
    "upward trend indicates consistent demand across all four quarters."
)

add_paragraph(
    content,
    "Operating profit was lowest in the first quarter at £54,000 and "
    "peaked in the fourth quarter at £85,000. The sharp rise in the "
    "final quarter reflects both higher revenue and a seasonal reduction "
    "in temporary staffing costs, which kept expenses below budget."
)

add_paragraph(
    content,
    "Across the full year, Northstar processed 4,800 orders, an average "
    "of 1,200 orders per quarter. Order volume was strongest in the fourth "
    "quarter and weakest in the first, matching typical retail demand "
    "patterns for office supplies."
)


# --------------------------------------------------
# 9. Top-selling products
# --------------------------------------------------

add_heading(content, "9. Top-Selling Products")

add_paragraph(
    content,
    "This section lists the products that generated the most revenue for "
    "Northstar in 2025. These items account for a large share of total "
    "sales and are closely monitored for stock availability. Management "
    "uses this list to plan purchasing and to negotiate better terms with "
    "suppliers."
)

products_data = [
    ["Product name", "Units sold", "Unit price", "Revenue"],
    ["A4 Copy Paper 500 sheets", "5,000", "£20", "£100,000"],
    ["HP Black Toner", "2,000", "£40", "£80,000"],
    ["Ballpoint Pen Black (12 pack)", "15,000", "£5", "£75,000"],
    ["Inkjet Cartridge", "1,500", "£40", "£60,000"],
    ["Notebook A5 80-page", "7,500", "£7", "£52,500"],
    ["Stapler Standard", "3,000", "£15", "£45,000"],
    ["Highlighters Set of 4", "4,500", "£10", "£45,000"],
    ["Desk Lamp LED", "1,000", "£30", "£30,000"],
]

add_table(content, products_data, [150, 75, 70, 75])

add_paragraph(
    content,
    "A4 copy paper was the top-selling product by revenue, moving 5,000 "
    "units for £100,000 in sales. Schools and small businesses drive most "
    "of this demand because they use large volumes of general-purpose "
    "copy paper throughout the year."
)

add_paragraph(
    content,
    "The HP Black Toner and Ballpoint Pen Black were the second and "
    "third best sellers respectively. Together they generated £155,000 in "
    "revenue. Printer supplies and writing instruments are among the most "
    "frequently reordered categories, making them reliable sources of "
    "repeat revenue."
)


# --------------------------------------------------
# 10. Operational metrics
# --------------------------------------------------

add_heading(content, "10. Fulfilment and Operations")

add_paragraph(
    content,
    "Northstar fulfilled all 4,800 orders during 2025 from its single "
    "warehouse in Manchester. The warehouse operates five days a week "
    "and employs twelve fulfilment staff. This section summarises the key "
    "operational metrics that management tracks each month."
)

fulfilment_data = [
    ["Metric", "2025 result", "2024 result", "Target"],
    ["Orders fulfilled", "4,800", "4,200", "5,000"],
    ["Average pick time (minutes)", "12.5", "14.2", "12.0"],
    ["On-time dispatch %", "97.5%", "94.8%", "98%"],
    ["Pick accuracy %", "98.2%", "96.5%", "98%"],
    ["Return rate %", "2.1%", "2.9%", "2%"],
    ["Average parcel weight (kg)", "2.4", "2.6", "2.5"],
]

add_table(content, fulfilment_data, [150, 90, 90, 70])

add_paragraph(
    content,
    "The warehouse fulfilled all 4,800 orders placed during 2025, an "
    "increase of 600 orders compared with the 4,200 orders fulfilled in "
    "2024. The average pick time improved from 14.2 minutes to 12.5 "
    "minutes per order, reflecting additional staff training and better "
    "warehouse layout."
)

add_paragraph(
    content,
    "On-time dispatch reached 97.5% in 2025, up from 94.8% in 2024. "
    "This metric measures the percentage of orders dispatched on or before "
    "the promised delivery date. Management considers it a key indicator "
    "of customer satisfaction because late deliveries are the most common "
    "source of customer complaints."
)

add_paragraph(
    content,
    "Pick accuracy, the percentage of orders packed without errors, "
    "improved to 98.2% in 2025 from 96.5% in 2024. The company reduced "
    "incorrect items sent to customers by introducing additional packing "
    "checks before parcels leave the warehouse, as noted in the company "
    "overview above."
)

add_paragraph(
    content,
    "The return rate fell from 2.9% in 2024 to 2.1% in 2025. Returns are "
    "caused primarily by damaged packaging, incorrect items, or customer "
    "change of mind. Management tracks the reasons for returns to identify "
    "areas where processes can be improved."
)


# --------------------------------------------------
# 11. Customer service
# --------------------------------------------------

add_heading(content, "11. Customer Service Metrics")

add_paragraph(
    content,
    "Customer service is provided through the company's website chat, "
    "email, and telephone lines. The customer service team consists of "
    "eight advisors who operate during UK business hours. This section "
    "reports the service levels achieved in 2025."
)

service_data = [
    ["Channel", "Volume 2025", "Response time", "Satisfaction"],
    ["Email", "1,440", "3.2 hours", "4.6 / 5"],
    ["Phone", "960", "28 seconds", "4.5 / 5"],
    ["Web chat", "720", "< 1 minute", "4.7 / 5"],
    ["In-person", "240", "Same day", "4.8 / 5"],
    ["Total", "3,360", "—", "4.6 / 5"],
]

add_table(content, service_data, [95, 95, 95, 90])

add_paragraph(
    content,
    "Email was the most used customer service channel in 2025, handling "
    "1,440 inquiries. The average response time was 3.2 hours, and the "
    "customer satisfaction rating was 4.6 out of 5. Most email inquiries "
    "related to order status and product availability."
)

add_paragraph(
    content,
    "Phone support handled 960 calls during the year with an average wait "
    "time of 28 seconds before an advisor answered. The satisfaction "
    "rating was 4.5 out of 5. Phone support is most frequently used for "
    "complex queries and returns processing."
)

add_paragraph(
    content,
    "Web chat, introduced in 2024, handled 720 conversations in 2025. "
    "Its average response time of less than one minute made it the "
    "fastest channel. It achieved the highest satisfaction rating at 4.7 "
    "out of 5, showing that quick responses improve the customer "
    "experience."
)

add_paragraph(
    content,
    "In-person support accounted for 240 interactions, mostly with local "
    "business customers who visited the warehouse to collect orders. "
    "It achieved the highest satisfaction rating of 4.8 out of 5, but "
    "the volume was too low to justify additional staffing investment."
)


# --------------------------------------------------
# 12. Marketing and advertising
# --------------------------------------------------

add_heading(content, "12. Marketing and Advertising")

add_paragraph(
    content,
    "Northstar spent £300,000 on marketing and advertising during 2025. "
    "This represents 12.5% of total revenue. The marketing budget is "
    "allocated across several channels to reach different customer "
    "segments and to maintain brand awareness throughout the year."
)

marketing_data = [
    ["Channel", "2025 spend", "Share", "ROI (est.)"],
    ["Digital advertising", "£180,000", "60%", "3.2x"],
    ["Print advertising", "£48,000", "16%", "1.8x"],
    ["Trade shows", "£36,000", "12%", "2.1x"],
    ["Content marketing", "£24,000", "8%", "2.5x"],
    ["Other", "£12,000", "4%", "1.5x"],
    ["Total", "£300,000", "100%", "—"],
]

add_table(content, marketing_data, [120, 90, 65, 70])

add_paragraph(
    content,
    "Digital advertising received the largest share of the marketing "
    "budget at £180,000, representing 60% of total spend. This channel "
    "focuses on Google and Facebook ads targeting small businesses and "
    "home offices. Management estimated its return on investment at 3.2 "
    "times the spend."
)

add_paragraph(
    content,
    "Print advertising accounted for £48,000 (16%) of the budget. Ads "
    "appeared in trade magazines read by procurement managers in schools "
    "and small businesses. The estimated return on investment was lower "
    "at 1.8 times, but the channel remains valuable for reaching "
    "business decision-makers who rarely use social media."
)

add_paragraph(
    content,
    "Trade shows, costing £36,000, allowed Northstar to meet customers "
    "face-to-face and demonstrate new products. The estimated return on "
    "investment was 2.1 times. The company attended eight regional and "
    "national exhibitions during 2025."
)


# --------------------------------------------------
# 13. Key metrics and calculations
# --------------------------------------------------

add_heading(content, "13. Key Metrics and Calculations")

add_paragraph(
    content,
    "This section summarises the most important figures in this report and "
    "explains how several key metrics are calculated. It is intended to help "
    "readers verify the results and understand the relationship between "
    "different figures mentioned in earlier sections."
)

calc_data = [
    ["Metric", "2025 value", "Calculation"],
    ["Revenue growth", "£300,000", "£2,400,000 - £2,100,000"],
    ["Revenue growth %", "14.3%", "£300,000 / £2,100,000"],
    ["COGS growth", "£156,000", "£1,416,000 - £1,260,000"],
    ["Gross profit margin", "41.0%", "£984,000 / £2,400,000"],
    ["Operating profit margin", "11.5%", "£276,000 / £2,400,000"],
    ["Operating profit growth", "£66,000", "£276,000 - £210,000"],
    ["Website + Email revenue", "£1,920,000", "£1,320,000 + £600,000"],
    ["Website + Email share", "80%", "£1,920,000 / £2,400,000"],
    ["Telephone vs Website AOV", "£61", "£143 - £82"],
]

add_table(content, calc_data, [150, 95, 130])

add_paragraph(
    content,
    "Revenue increased by £300,000 from 2024 to 2025, rising from "
    "£2,100,000 to £2,400,000. This represents a percentage increase of "
    "approximately 14.3%."
)

add_paragraph(
    content,
    "Cost of goods sold increased by £156,000, rising from £1,260,000 "
    "in 2024 to £1,416,000 in 2025. The gross profit for 2025 was "
    "£984,000, giving a gross profit margin of 41.0%."
)

add_paragraph(
    content,
    "Operating profit increased by £66,000 from £210,000 in 2024 to "
    "£276,000 in 2025. The operating profit margin was 11.5% of revenue, "
    "calculated as £276,000 divided by £2,400,000."
)

add_paragraph(
    content,
    "Combined revenue from the Website and Email channels was £1,920,000, "
    "representing 80% of the company's total 2025 revenue. The Website "
    "channel alone generated £1,320,000 (55%) and the Email channel "
    "generated £600,000 (25%)."
)

add_paragraph(
    content,
    "The Telephone channel's average order value of £143 was £61 higher "
    "than the Website channel's average order value of £82. This difference "
    "reflects the fact that telephone customers place fewer but larger "
    "orders, often with assistance from customer service staff."
)


# --------------------------------------------------
# Generate PDF
# --------------------------------------------------

pdf.build(
    content,
    onFirstPage=add_page_number,
    onLaterPages=add_page_number,
)

print(f"PDF created successfully: {OUTPUT_FILE}")
