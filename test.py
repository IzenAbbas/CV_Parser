import pandas as pd

df=pd.read_csv('Final.csv')
print(df['Category'].value_counts())


job_role_mapping = {
    # ===== CORE SOFTWARE DEVELOPMENT =====
    "Java Developer": [
        "Java Developer", "J2EE Developer", "Java & Devops Developer", "Java AWS Developer",
        "Java Architect", "Java Backend Developer", "Java Developer Intern", "Java Engineer",
        "Java Full Stack Developer", "Java Full stack developer", "Java Full-Stack Developer",
        "Java Lead Developer", "Java Programmer", "Java Programmer Analyst", "Java Senior Developer",
        "Java Software Developer", "Java Software Engineer", "Java Tech Lead", "Java Technical Lead",
        "Java UI Developer", "Java Web Application Developer", "Java Web Developer", "Java developer",
        "Java software engineer", "Java developer intern", "Java/J2EE Developer", "Java/J2EE developer",
        "Java/AWS developer", "Java/DevOps Engineer", "Java/J2EE Full Stack Developer",
        "Java/J2EE developer", "Java/J2EE/UI Developer (Full Stack Developer)", "Java/J2ee Developer",
        "Senior Java Developer", "Senior Java Developer (FullStack)", "Senior Java Full Stack Developer",
        "Senior Java Fullstack Developer", "Senior Java J2EE Developer", "Senior Java developer",
        "Senior Java/J2EE Developer", "Sr Java Developer", "Sr Java Full Stack Developer",
        "Sr Java Fullstack Developer", "Sr Java/J2EE Developer", "Sr. JAVA Developer",
        "Sr. JAVA/J2EE Developer & Full Stack Developer", "Sr. Java / J2EE Developer",
        "Sr. Java AWS developer", "Sr. Java Developer", "Sr. Java developer", "Sr. Java/J2EE Developer",
        "Sr. Java/J2EE developer", "Sr. Java/J2EE Full Stack Developer", "Sr. Java/J2ee Developer",
        "Sr. Java/Liferay Portal Developer", "Lead Java Developer", "Lead Developer",
        "Lead Java Developer", "Associate Java product developer", "Associate Java/J2EE developer",
        "FullStack Java developer", "Sr Full Stack Java Developer", "Sr FullStack Java developer",
        "Jr. Java Developer", "Junior Java Developer", "Junior Java developer", "Core/Server Side Java Developer",
        "CoreJava Developer", "Full Stack Java Developer", "Full Stack Java Developer Intern",
        "Full Stack Java EE Developer", "Full Stack Java developer", "Full Stack Java/J2EE Developer",
        "Full-Stack Java Developer", "Full-stack Java Developer", "Java developer intern",
        "Java development tech lead", "Sr. FullStack Java Developer", "Backend Java Developer",
        "Java Backend Developer", "Java backend developer", "Senior Associate Full Stack Java developer",
        "Sr. Full Stack Java Developer", "Sr. Full Stack Java developer", "Lead/Sr JAVA Developer",
        "Lead/Sr. Java Developer", "Sr Full Stack Java Developer", "Associate Full-Stack Java Developer"
    ],

    "Python Developer": [
        "Python Developer", "Python AWS Developer", "Python Data Analyst", "Python Full Stack Developer",
        "Python Fullstack Developer", "Python Programmer", "Python Software Developer", "Python Tutor",
        "Python developer", "Python backend developer", "Python developer and API engineer", "Python programmer",
        "Python/Django Developer", "Backend Python developer and occasional Go developer", "Sr. Python Developer",
        "Sr. Python / AWS Developer", "Sr. Python Full Stack Developer", "Sr. Python developer",
        "Sr. Python/Django Developer", "Sr Python Developer", "Sr Python Full Stack Developer",
        "Senior Python Developer", "Senior Python Full Stack Developer", "Senior Python Software Engineer",
        "Lead Python Developer", "Python & Java Full Stack Developer", "Python & Webdeveloper",
        "Python / Full Stack Developer", "Python API Engineer", "Python Backend Developer",
        "Python Developer - Machine Learning", "Python Developer Hadoop",
        "Python Developer/ Machine Learning Engineer/ Data Scientist",
        "Python Developer/Analyst", "Python Development Instructor", "Python Django Backend Web Developer",
        "Python Unit Tester", "Backend Python developer", "Full Stack Python Developer",
        "Full Stack Python/Django Developer", "Full Stack Python/React Developer", "Jr. Python Developer",
        "Angular/ Python developer", "AWS/ Python Developer", "Sr. Python Developer - Data Engineering",
        "Sr. Python Developer/AWS Engineer", "Sr. Python Full Stack Developer", "Sr. Python/Django Developer",
        "Sr. Python Full Stack Developer", "Sr. Python developer", "Sr. Python/Django Developer",
        "Golang/ Python Developer", "Knowledge Base Python Developer", "Sr. Python Application Developer"
    ],

    "JavaScript & Web Developer": [
        "JavaScript Developer", "JavaScript Instructor", "JavaScript Web Developer Intern",
        "Javascript Engineer", "ReactJS Developer", "Reactjs developer", "React Developer",
        "React JS", "React JS developer", "React UI/ Front End Developer", "React/ Python developer",
        "React Native Developer", "Senior React Native Developer", "Senior ReactJS Developer",
        "Sr. React Software Developer", "Sr. React Native Developer", "Sr. ReactJS Developer",
        "Node BackEnd Developer", "Angular Developer", "Angular Node Postgre Developer",
        "Angular/ Java Developer", "Angular2", "Angular2 Software Developer", "AngularJS Developer",
        "AngularJS/UI/ Java Developer/Lead", "Angular 2/4/ Front- End Developer", "Sr. Angular Developer",
        "Sr. Angular JS Developer", "Sr. Angular Web Developer", "Sr. UI/Angular developer",
        "Sr. UI/ Front- End- Developer/ Mean Stack Developer", "MEAN Stack Developer",
        "Mean Stack Developer", "Mean Stack Developer/UI Developer", "Full Stack Angular Developer",
        "Full Stack MEAN Stack Developer", "AngularJS / Front-End Developer", "Sr. Angular JS Developer",
        "Vue.js Developer", "Front end Vue.js Developer", "Spark React Developer"
    ],

    "ASP.NET & C# Developer": [
        ".NET Software Developer", ".Net Developer", "C# .NET Backend Developer", "C# / ASP.NET Developer",
        "C# / Unity3D developer", "ASP.Net Web Developer", "DotNet Developer", "Sr .Full Stack Java Developer",
        "Sr. C#/.Net Developer", "Sr. Net Software Engineer", "Sr.Net Software Engineer",
        "Sr. NET Developer", "Senior .NET consultant"
    ],

    "Full Stack Developer": [
        "Full Stack Developer", "Full Stack Developer ", "Full Stack Developer & Digital Marketing Specialist",
        "Full Stack Developer - Child Welfare Intake Referral Portal",
        "Full Stack Developer - Front End and Back End developer",
        "Full Stack Developer / UI Lead / UI Architect", "Full Stack Engineer", "Full Stack Engineer Intern",
        "Full Stack JAVA Developer", "Full Stack Java Consultant", "Full Stack Technical Architect",
        "Full Stack Web Developer", "Full Stack web developer", "Full stack Developer", "Full stack developer",
        "Full stack software developer", "Full-Stack Developer", "Full-Stack Developer WordPress",
        "Full-Stack Engineer/ Database Administrator", "Full-Stack Web Developer", "Full-stack Developer",
        "FullStack Developer", "Full/ MEAN Stack Developer", "BACKEND DEVELOPER",
        "FULL STACK CLOUD ENGINEER - BOOTCAMP", "FULL STACK JAVA DEVELOPER", "FULL-STACK DEVELOPER",
        "Sr. Full Stack Python Developer", "Sr. Full Stack developer- IVR Developer", "Sr. Full Stack UI Developer"
    ],

    "Frontend Developer": [
        "Front End Developer", "Front End Developer Intern", "Front- End Developer",
        "Front-End Developer", "Front-End Developer Intern", "Frontend Developer", "Frontend Engineer",
        "Frontend Web Developer", "Frontend UI Developer", "Frontend developer", "Frontend developer/lead of front-end",
        "FRONT END DEVELOPER", "FRONT END DEVELOPER / UIUX DESIGNER", "FRONT END DEVELOPER AND DESIGNER",
        "FRONT END DEVELOPER CONSULTANT", "FRONT END DEVELOPER/IT SUPPORT", "FRONT END WEB DEVELOPER",
        "FRONT END WEB DEVELOPER/DESIGNER", "FRONT- END DEVELOPER", "FRONT- END DEVELOPER (FREELANCE)",
        "FRONT- END DEVELOPER / DESIGNER", "FRONT- END DEVELOPER / UX DESIGNER", "FRONT- END WEB DEVELOPER",
        "FRONT-END DEVELOPER", "FRONT-END DEVELOPER INTERN", "JR FRONT END WEB DEVELOPER",
        "SENIOR DESIGNER / FRONT-END DEVELOPER", "SENIOR FRONTEND WEB DEVELOPER", "SENIOR FRONT- END DEVELOPER",
        "SENIOR FRONT END WEB DEVELOPER", "SENIOR WEB DEVELOPER", "SENIOR WEB DESIGNER/ FRONT- END DEVELOPER",
        "Lead Front End Developer", "Lead Front End Developer - Designer - Project Manager",
        "Lead Front End Software Engineer",
        "Lead Front End UI Engineer", "Lead Front End Web Developer", "Lead Front- End Engineer",
        "Lead Front- end Developer", "Lead Front-End Developer", "Lead Front-End Engineer",
        "Lead Front-End Web Developer",
        "Lead Front-end Developer", "Associate Front-End Developer", "Associate Email Developer",
        "Associate Experience Designer"
    ],

    "Backend Developer": [
        "Back End Developer", "Back End Developer & Project Manager - Systems Integration School Project",
        "Back End Web Developer", "Back end developer", "Back-End Developer | Contract", "Back-end Developer",
        "Back-end Java developer", "Backend Developer", "Backend Engineer (Scala)", "Backend Java Developer",
        "Backend Python developer and occasional Go developer", "Backend Software Engineer", "Backend Web Developer",
        "Backend/ Full Stack Web Developer", "Server-Side Developer", "Node.js Developer",
        "Express.js Developer"
    ],

    "Mobile Developer": [
        "Android Developer", "Android Developer Intern", "Android Developer/Architect", "Android developer",
        "Android/ Mobile Application Developer", "Android Application Developer", "Android Applications Developer",
        "Android Backend Developer", "iOS Developer", "iOS Engineer", "iOS SOFTWARE ENGINEER",
        "Senior IOS Developer", "Sr. IOS Developer", "Mobile App Developer", "Mobile Developer",
        "Mobile Device Management Project Lead & IT Technician", "Mobile Device Manager", "Mobile Engineer Tier",
        "Mobile Programming - Software Developer", "Mobile Tour Guide App", "Mobile/Web App Developer",
        "Sr. Mobile Developer", "App Developer & Retail Investor", "AR/VR Unity Developer (C#)",
        "Game Developer", "Game Developer Intern", "Unity3D Developer"
    ],

    "Web Designer": [
        "Web Designer", "Web Designer & Developer (Freelance)", "Web Designer & Front-end Developer",
        "Web Designer (Front-End developer)", "Web Designer / Developer", "Web Designer / Front End Developer",
        "Web Designer / Front-End Developer", "Web Designer / Front-end Developer",
        "Web Designer / Store Owner/Manager",
        "Web Designer / UI Developer", "Web Designer and Developer", "Web Designer | Front End Developer",
        "Web Designer/ Developer", "Web Designer/ Front End Developer", "Web Designer/ Front-end Developer II",
        "Web Designer/Developer", "Web Designer/UI Designer/Lead Front-End Drupal Web Developer",
        "Web Designing", "Designer", "Design Integrator ( Front End Developer)", "UX Designer / Front End Developer"
    ],

    "UI/UX Designer": [
        "UX Designer", "UX Designer / Front End Developer", "UX Designer / Front-End Developer",
        "UX Designer and Front End Developer", "UX Engineer", "UX Manager", "UX Research and Design Lead",
        "UX Researcher & Designer", "UX/UI Designer", "UX/UI Designer & Developer - Freelance Full-time",
        "UX/UI Designer & Project Manager", "UX/UI Developer", "UX/UI Developer & Designer",
        "UX/UI Front End Developer", "UX/UI/ Front- end/Web Developer", "UX/UI/Web Designer",
        "UI Designer and Developer", "UI Designer/ Developer", "UI Designer/ Web Production Specialist",
        "UI designer", "UI developer", "STAFF UX DESIGNER", "Art Director • UX/UI Designer",
        "Chief Digital Officer", "Creative Director", "UX / UI Design / Graphic Design", "UX Architect / Co-Founder"
    ],

    "Graphic & Multimedia Designer": [
        "Graphic Designer", "Graphic Designer & UI/UX", "Graphic Designer + Front-End Developer",
        "Graphic Designer / Front End Developer", "Graphic Designer and Front End Developer",
        "Graphic Designer and Front-End Developer", "Graphic Designer/ Project Manager", "Graphic and Web Designer",
        "Graphic designer", "Graphic/ Web Designer", "Multimedia Designer", "Multimedia Producer",
        "Multimedia Architect", "Photographer", "Photographer & Sales", "Photographer/Film Editor",
        "Photographer/Sales Associate", "Photographer/Videographer/Marketer", "Graphics Designer",
        "Creative Director & Founder", "Creative Artist and Developer"
    ],

    "Database Administrator": [
        "DATABASE ADMINISTRATOR", "DATABASE ADMINISTRATOR / ANALYST", "DATABASE ADMINISTRATOR / DEVELOPER",
        "DATABASE ADMINISTRATOR AND AWS CLOUD AMINISTRATOR", "DATABASE ADMINISTRATOR II",
        "DATABASE ADMINISTRATOR/CNC ADMINISTRATOR", "DATABASE ADMINISTRATOR/Data Analyst",
        "DATABASE ANALYST/ ADMINISTRATOR", "DATABASE ANALYST/ PROGRAMMER",
        "DATABASE AND SYSADMIN CONSULTANT", "DATABASE DEVELOPER", "DATABASE DEVELOPER - - PROJECT MANAGER - IT MANAGER",
        "DATABASE DEVELOPER- ADMINISTRATOR", "DATABASE ENGINEER TIER 3", "DATABASE MASTER ADMINISTRATOR",
        "DATABASE SOLUTION ARCHITECT", "DBA", "DBA / Developer", "DBA Consultant", "DBA Engineer",
        "DBA Reporting Manager", "Database Administrator", "Database Administrator & Academic Support Specialist",
        "Database Administrator & System Analyst (contractor)", "Database Administrator & Web Developer",
        "Database Administrator & Webmaster", "Database Administrator (Active TS/SCI)",
        "Database Administrator (Consultant)",
        "Database Administrator (Contractor)", "Database Administrator (DBA)",
        "Database Administrator (Graduate Assistantship)",
        "Database Administrator (Oracle)", "Database Administrator (Part-Time)", "Database Administrator (SQL)",
        "Database Administrator (Team Lead)", "Database Administrator (contractor)",
        "Database Administrator - IT Support Specialist", "Database Administrator -MSI",
        "Database Administrator / Assistant Controller", "Database Administrator / Database Developer",
        "Database Administrator / Developer", "Database Administrator / Network Admin",
        "Database Administrator / Software Developer", "Database Administrator / Technical Support",
        "Database Administrator / Viticulturalist / Geospatial Analyst",
        "Database Administrator /Application Support Specialist",
        "Database Administrator 3", "Database Administrator Associate", "Database Administrator Consultant",
        "Database Administrator Data Modeler", "Database Administrator I", "Database Administrator I/ Sysadmin",
        "Database Administrator II", "Database Administrator III", "Database Administrator Intern",
        "Database Administrator Manager", "Database Administrator Specialist", "Database Administrator and Analyst",
        "Database Administrator internship", "Database Administrator(DBA)", "Database Administrator- Full-time",
        "Database Administrator-Contractor", "Database Administrator/ Database Developer",
        "Database Administrator/ Financial Budget Analyst", "Database Administrator/ IT Manager",
        "Database Administrator/ IT Support", "Database Administrator/ Reporting Analyst",
        "Database Administrator/ System Analyst", "Database Administrator/AWS Architect",
        "Database Administrator/Analyst", "Database Administrator/DBA", "Database Administrator/DevOps",
        "Database Administrator/Developer", "Database Administrator/Development Lead",
        "Database Administrator/Engineer", "Database Administrator/GIS solutions",
        "Database Administrator/IT Assistant",
        "Database Administrator/Legal Secretary", "Database Administrator/Maintenance Manager",
        "Database Administrator/Merchandise Coordinator", "Database Administrator/Programmer Analyst",
        "Database Administrator/Receptionist", "Database Administrator/Senior Consultant",
        "Database Administrator/Software Engineer", "Database Administrator/Sourcer",
        "Database Administrator/System Engineer", "Database Administrator/Web Developer",
        "Database Administrator/programmer", "Database Administrator; Developer; System Analyst"
    ],

    "Database Developer": [
        "Database Developer", "Database Developer / Database Admin II", "Database Developer and Admin",
        "Database Developer/ Administrator", "Database Developer/ Business Intelligence (BI)",
        "Database Developer/Analyst", "Database Developer/DBA", "Database Engineer", "Database Engineer Senior",
        "Database Lead", "Database Programmer", "Database Software Engineer",
        "Database Solution Architect AWS Oracle RDS Databases",
        "Database Specialist", "Database Specialist II", "Database Specialist and User Support Analyst",
        "Database Support Service Lead", "Database System Administrator", "Database administrator",
        "Database administrator at client side", "Database administrator-1", "Database administrator/Devops Engineer",
        "Database administrator/Sales tax specialist"
    ],

    "Database Analyst": [
        "Database Analyst", "Database Analyst /System Administrator", "Database Analyst III",
        "Database Analyst Princ", "Database Analyst and Administrator", "Database Analyst and ETL Developer",
        "Database Analyst and System Administrator", "Database Analyst/Specialist", "Senior Database Analyst",
        "Senior Database Analyst/ Administrator", "Advanced Database Analyst/Programmer (Remote)",
        "Associate Database Administrator Z/OS", "Associate Database administrator",
        "Associate Database/Systems Administrator"
    ],

    "Data Scientist & Analytics": [
        "Data Scientist", "Data Scientist Intern", "Data Scientist intern", "Data Scientist/ Machine Learning",
        "Data Scientist/Analytics Consultant", "Data Scientist/Data Science Manager", "Data Scientist/Data engineer",
        "Senior Data Scientist", "Data Analyst", "Data Analyst & Database Administrator", "Data Analyst - Excel/SQL",
        "Data Analyst / Java Developer", "Data Analyst / Project Manager", "Data Analyst Intern",
        "Data Analyst Student Worker", "Data Analyst and ERP Database Administrator",
        "Data Analyst and Reporting - National Call Center Operations", "Data Analyst | IT Support Training Leader",
        "Data Analyst/ Python Developer", "Data Analytics and Reporting Analyst", "Data Annotation Specialist",
        "Business Data Analyst", "Data & Ecommerce Analyst", "Data Analysis Intern",
        "Senior Data Analyst", "Machine Learning Scientist"
    ],

    "Business Analyst": [
        "Business Analyst", "Business Analyst - Business Development", "Business Analyst - IT Projects",
        "Business Analyst / Project Manager", "Business Analyst Consultant", "Business Analyst I",
        "Business Analyst II/ Data Architect", "Business Analyst III", "Business Analyst IV",
        "Business Analyst Intern", "Business Analyst Project Coordinator", "Business Analyst Salesforce/Pardot",
        "Business Analyst V", "Business Analyst for People Analytics", "Business Analyst/ IT Consultant",
        "Business Analyst/ Project Manager", "Business Analyst/ Scrum Master",
        "Sr Business Analyst - Test Management and Change Management",
        "Sr Cyber Security Associate", "Senior Business Analyst",
        "Senior Business Analyst / Reengineering / Project Manager",
        "Senior Business Analyst/ Project Manager",
        "Senior Business Intelligence Analyst and Database Developer (Team Lead)",
        "Senior Business Process Analyst", "Senior Business Process Analyst / Project Manager",
        "Senior Business Systems Analyst"
    ],

    "Systems Administrator": [
        "System Administrator", "System Administrator - Contract", "System Administrator - Help Desk Level II",
        "System Administrator - Network", "System Administrator / Deskside support", "System Administrator Forensics",
        "System Administrator I", "System Administrator II", "System Administrator II- Indrasoft",
        "System Administrator Sr", "System Administrator Tier 2", "System Administrator V - Network engineer",
        "System Administrator and Computer Technician", "System Administrator and Operations Manager",
        "System Administrator- IT Support Analyst V", "System Administrator/ Network Administrator",
        "System Administrator/ Network Engineer", "System Administrator/ Security Analyst",
        "Systems Administrator", "Systems Administrator - Wintel Server Admin",
        "Systems Administrator / Data Center Technician", "Systems Administrator / IT Specialist",
        "Systems Administrator / Network Security Support", "Systems Administrator / Project Manager",
        "Systems Administrator / Project Manager / IT Generalist", "Systems Administrator / SharePoint Administrator",
        "Systems Administrator 2", "Systems Administrator II", "Systems Administrator III",
        "Systems Administrator Intermediate", "Systems Administrator Professional",
        "Systems Administrator/ Database Application Manager", "Systems Administrator/ Network Administrator",
        "Systems Administrator/ Server Engineer", "Systems Administrator/ Systems Analyst",
        "Systems Administrator/Cyber Security Engineer", "Systems Administrator/Director of IT",
        "Systems Administrator/Migration", "Systems Administrator/Systems Analyst"
    ],

    "Network Administrator": [
        "NETWORK ADMINISTRATOR", "NETWORK ADMINISTRATOR / IT SUPPORT TECHNICIAN",
        "NETWORK ADMINISTRATOR / PRODUCTION & VINYL APPLICATION LEAD", "NETWORK ADMINISTRATOR COMPETENCE",
        "Network Administrator", "Network Administrator & Director of IT", "Network Administrator & Program Manager",
        "Network Administrator & Security Officer", "Network Administrator & VOIP Administrator",
        "Network Administrator (On Call)", "Network Administrator (Promotion)", "Network Administrator - Level 2",
        "Network Administrator - Systems Operations", "Network Administrator - Team Lead (Contract)",
        "Network Administrator / Desktop Support", "Network Administrator / Desktop Technician",
        "Network Administrator / General Manager", "Network Administrator / Help Desk Technician",
        "Network Administrator / Helpdesk Manager", "Network Administrator / IT Security Analyst",
        "Network Administrator / Interim IT Director (past 2 Months)",
        "Network Administrator / Mission Defense Analyst",
        "Network Administrator / Network analyst", "Network Administrator / Project Manager",
        "Network Administrator / Site Supervisor / IT Support", "Network Administrator / Systems Administrator"
    ],

    "Network Engineer": [
        "NETWORK ENGINEER", "NETWORK ENGINEER (GRADUATE ASSISTANT)", "NETWORK ENGINEER/IT ADMINISTRATOR",
        "Network Engineer", "Network Engineer & Administrator", "Network Engineer & Business Consultant",
        "Network Engineer (By contract)", "Network Engineer (Consultant)", "Network Engineer (MSOC III)",
        "Network Engineer (Remote)", "Network Engineer (remote/on-premises)", "Network Engineer - Tier 3",
        "Network Engineer / IT Contractor", "Network Engineer / Information Security Engineer",
        "Network Engineer / Network Administrator", "Network Engineer / Programmer",
        "Network Engineer / Systems Administrator",
        "Network Engineer Contractor", "Network Engineer II", "Network Engineer II - Contract",
        "Network Engineer III", "Network Engineer Intern", "Network Engineer Virtualization",
        "Network Engineer and Software Developer", "Network Engineer/ Administrator", "Network Engineer/ IT Consultant",
        "Network Engineer/ IT Consultant/ IT Manager", "Network Engineer/ operation support",
        "Network Engineer/Project Manager", "Network Engineer/Project Manager/Cloud Solution Architect",
        "Network Engineer/Systems Administrator", "Senior Network Engineer", "Senior Network Engineer - Advisor"
    ],

    "Cybersecurity Analyst": [
        "CYBER SECURITY ANALYST", "CYBER- SECURITY ANALYST", "Cyber Security Analyst",
        "Cyber Security Analyst (Intern)", "Cyber Security Analyst (Team Lead)", "Cyber Security Analyst - Contractor",
        "Cyber Security Analyst - Rx- IT Security Assessment", "Cyber Security Analyst - SOC",
        "Cyber Security Analyst - Senior", "Cyber Security Analyst I", "Cyber Security Analyst II",
        "Cyber Security Analyst II (Q - DoE Security Clearance)", "Cyber Security Analyst III",
        "Cyber Security Analyst IT", "Cyber Security Analyst Intern", "Cyber Security Analyst MID",
        "Cyber Security Analyst Support Center", "Cyber Security Analyst/ IT Auditor",
        "Cyber Security Analyst/Engineer", "Cyber Security Analyst/Identity Protection & Management",
        "Cyber Security Analyst/Incident Response", "Cyber Security Analyst/RMF Specialist (Veteran Affairs)",
        "Cybersecurity Analyst", "Cybersecurity Analyst (Remote)",
        "Cybersecurity Analyst/Information Systems Security Officer (ISSO)",
        "Cybersecurity Analyst/Instructor", "Cybersecurity Consultant", "Sr. Cyber Security Analyst",
        "Sr. Cyber Security Engineer",
        "Sr. Cyber Security Policy & Compliance Analyst / Information System Security Engineer",
        "Sr. Cyber Security Project Manager", "Sr. Cyber Security Reviewer", "Senior Cyber Security Analyst",
        "Senior Cyber Security Engineer (Tier 3 SOC)", "Senior Cyber Security Engineer II/ISSO"
    ],

    "Information Security Analyst": [
        "INFORMATION SECURITY ANALYST", "Information Security Analyst",
        "Information Security Analyst & Pre/Post sales Engineer",
        "Information Security Analyst (Contract)", "Information Security Analyst (Contractor)",
        "Information Security Analyst (RGC) Contractor", "Information Security Analyst - Cloud Security",
        "Information Security Analyst - Cybersecurity Division", "Information Security Analyst - Intermediate",
        "Information Security Analyst - Network Administrator", "Information Security Analyst - Senior IT Auditor",
        "Information Security Analyst - Team Lead", "Information Security Analyst - Vulnerability Program Manager",
        "Information Security Analyst / Database Admin", "Information Security Analyst / Engineer",
        "Information Security Analyst /Cloud Administrator", "Information Security Analyst 3",
        "Information Security Analyst DLP Analyst", "Information Security Analyst I",
        "Information Security Analyst II", "Information Security Analyst Intern", "Information Security Analyst V",
        "Information Security Analyst team lead", "Information Security Analyst- Enterprise Systems",
        "Information Security Analyst/ Risk Analyst", "Information Security Analyst/Compliance",
        "Information Security Analyst/Engineer"
    ],

    "IT Security Engineer": [
        "IT SECURITY ANALYST", "IT SECURITY ANALYST (Contract)", "IT SECURITY ANALYST (Full-Time Internship)",
        "IT SECURITY ANALYST (SOC)", "IT SECURITY ANALYST CYBER SECURITY OPS", "IT SECURITY ENGINEER",
        "IT SECURITY SPECIALIST", "IT SECURITY/APPLICATIONS ANALYST", "IT Security Analyst",
        "IT Security Analyst & Key Administrator", "IT Security Analyst (Contractor)", "IT Security Analyst (GRC)",
        "IT Security Analyst (ISSO)", "IT Security Analyst (Senior)", "IT Security Analyst - Associate",
        "IT Security Analyst - Full Time", "IT Security Analyst - Threat and Vulnerability Management",
        "IT Security Analyst Apprentice", "IT Security Analyst Compliance & Assurance", "IT Security Analyst I",
        "IT Security Analyst II", "IT Security Analyst II | Software Engineer", "IT Security Analyst III",
        "IT Security Analyst IV", "IT Security Analyst Intern", "IT Security Analyst Sr",
        "IT Security Analyst and Business Dev", "IT Security Analyst and Support", "IT Security Analyst-CA-MMIS",
        "IT Security Analyst-Consultant", "IT Security Analyst. Strategic Team", "IT Security Analyst/ Assessor",
        "IT Security Analyst/ IT Security Metrics", "IT Security Analyst/ Security Control Assessor",
        "IT Security Analyst/A&A Specialist", "IT Security Analyst/Compliance", "IT Security Analyst/Consultant",
        "IT Security Analyst/Engineer", "IT Security Analyst/Help Desk Technician",
        "IT Security Analyst/Vulnerability Management", "IT Security Analyst/Vulnerability Management Lead"
    ],

    "Security Operations & Incident Response": [
        "SOC Analyst", "SOC Computer Security Incident Response Analyst", "SOC Manager", "SOC Team Lead",
        "Security Operations Analyst", "Security Operations Analyst (SOC)", "Security Operations Center (SOC) Analyst",
        "Security Operations Engineer", "Security Operations Manager", "Security Operations Specialist II",
        "Security Ops Center IT Security Analyst", "Incident Response", "Incident Response Specialist",
        "Incident/Queue Manager", "Cyber Intrusion Detection Analyst",
        "Intrusion Detection Analyst - Security Operations Center"
    ],

    "Cloud & DevOps Engineer": [
        "AWS Administrator", "AWS Architect / Senior Database Administrator", "AWS Cloud Architect",
        "AWS DEVELOPER", "AWS DevOps Engineer", "AWS Engineer", "AWS Solution Architect Associate",
        "AWS Solution Architect/DevOps Engineer", "AWS Solutions Architect", "AWS Solutions Architect / Cloud Engineer",
        "AWS Web Application Firewall Engineer", "Azure Cloud Engagement Manager (Consultant)",
        "Azure DevOps Administrator - Consultant", "Azure DevOps Engineer", "Azure Engineer",
        "Azure Security and Identity Support Specialist / Software Engineer", "Azure VM Admin",
        "Cloud Administrator", "Cloud Administrator I", "Cloud Analyst", "Cloud Architect",
        "Cloud Cybersecurity Engineer", "Cloud Database Administrator", "Cloud Engineer",
        "Cloud Network Engineer", "Cloud Operations Analyst / Developer", "Cloud Operations Engineer Intern",
        "Cloud Operations Lead", "Cloud Project Manager/Lead Scrum Master", "Cloud Security Architect/Engineer",
        "Cloud Security Engineer", "Cloud Security Support Engineer", "Cloud Services Manager",
        "Cloud Solutions Architect", "Cloud Support Engineer", "Cloud networking support engineer",
        "DevOps", "DevOps ( Front End Developer)", "DevOps Cloud Architect", "DevOps DBA",
        "DevOps Engineer", "DevOps Engineer - (Technical Consultant)", "DevOps Engineer/AWS",
        "DevOps Lead", "DevOps Manager", "DevOps/Middleware Developer and System Integrations",
        "Devops Engineer", "Senior DevOps Engineer", "Sr. DevOps Engineer", "Sr. DevOps/Cloud Engineer"
    ],

    "IT Project Manager": [
        "IT PROJECT MANAGER", "IT PROJECT MANAGER (Global Projects)", "IT PROJECT MANAGER (Healthcare/Financial/Legal)",
        "IT PROJECT MANAGER (REMOTE)", "IT PROJECT MANAGER / SME", "IT PROJECT MANAGER CONNECTED CARE",
        "IT PROJECT MANAGER II", "IT PROJECT MANAGER PRIVACY & SECURITY GOVERNANCE",
        "IT PROJECT/FACILITIES MANAGER", "IT Project Manager", "IT Project Manager & Database Developer",
        "IT Project Manager & Managing Partner", "IT Project Manager & Scrum Master", "IT Project Manager ( Contract )",
        "IT Project Manager (Communications Officer)", "IT Project Manager (Consultant)",
        "IT Project Manager (Contract)",
        "IT Project Manager (Contractor)", "IT Project Manager (Financial and Planning)",
        "IT Project Manager (Lead) - Consultant",
        "IT Project Manager (TPM)", "IT Project Manager (contract)", "IT Project Manager (contractor)",
        "IT Project Manager (remote)", "IT Project Manager - AMAS Region", "IT Project Manager - BI Reports",
        "IT Project Manager - Change Management Department", "IT Project Manager - Cloud Infrastructure",
        "IT Project Manager - Contingent", "IT Project Manager - Contract", "IT Project Manager - Contractor",
        "IT Project Manager - Enterprise Technologies", "IT Project Manager - Managed Network Services",
        "IT Project Manager - Supply Chain", "IT Project Manager - Team Lead", "IT Project Manager - contract"
    ],

    "IT Program/Portfolio Manager": [
        "IT Program Manager", "IT Program Manager - Senior Consultant",
        "IT Program Manager / Portfolio Leader (Remote)",
        "IT Program Manager III - PMO Office", "IT Program Manager Senior", "IT Program Manager and IT Project Manager",
        "IT Program Manager-consultant", "IT Program Manager/Sr. IT Project Manager", "IT Program and PMO Manager",
        "IT Program manager", "IT Program/ Manager", "Program Manager", "Program Manager & Project Manager III",
        "Program Manager ( IT Security Risk Compliance)", "Program Manager - Engineering Security",
        "Program Manager - Finance IT group (contract through Randstad Tech)", "Program Manager - Risk Operations",
        "Program Manager / Project Manager", "Program Manager IV", "Program Manager and Content Developer",
        "Program Manager for Workday ERP Implementation", "Program Manager/ Project Manager"
    ],

    "Scrum Master & Agile Coach": [
        "AGILE SCRUM MASTER/COACH", "SCRUM MASTER", "Agile Coach", "Agile Coach / Scrum Master",
        "Agile Coach/Transformation Lead Consultant - IT Security", "Agile Full-Stack Delivery Lead",
        "Agile Project Manager/Associate Scrum Master", "Agile Scrum Master",
        "Agile Scrum Master and Mobile Project Manager",
        "Agile Team Lead", "Scrum Master", "Scrum Master - Executive Support Team", "Scrum Master Contractor",
        "Scrum Master/ Business Analyst", "Scrum Master/ IT Project Manager",
        "Scrum Master/ Oracle Database Engineer (Sr Analyst)",
        "Scrum Master/ Project Manager", "Scrum Master/ Senior IT P.M./R.T.E", "Scrum Master/Agile",
        "Scrum Master/Agile Coach/Project Manager", "Scrum Master/IT Project Manager", "Scrum Master/Project manager"
    ],

    "IT Help Desk & Support": [
        "Help Desk", "Help Desk Administrator", "Help Desk Administrator/IT Assistant to IT Manager",
        "Help Desk Analyst", "Help Desk Assistant - Contractor", "Help Desk Lead", "Help Desk Manager",
        "Help Desk Representative", "Help Desk Representative II", "Help Desk Specialist", "Help Desk Specialist I",
        "Help Desk Supervisor", "Help Desk Support Analyst", "Help Desk Support Specialist",
        "Help Desk Technical Support", "Help Desk Technician", "Help Desk Technician Specialist",
        "Help Desk/Desktop Support Technician", "Help Desk/System/ Network Administrator",
        "Help desk Analyst Customer Support", "Helpdesk /IT Network Specialist", "Helpdesk Analyst",
        "Helpdesk Computer Technician", "Helpdesk Specialist", "Helpdesk Support Analyst",
        "Helpdesk Support Technician", "Helpdesk Technician", "Helpdesk Technician/ Network Consultant",
        "Helpdesk/Application Support", "Helpdesk/Desktop Support Specialist"
    ],

    "Desktop Support Technician": [
        "Desktop Support", "Desktop Support / Network Administrator", "Desktop Support Analyst",
        "Desktop Support Engineer", "Desktop Support Intern", "Desktop Support Specialist",
        "Desktop Support Specialist (contract)", "Desktop Support Technician", "Desktop Support/ Network Administrator",
        "Desktop Support/Office 365 Administrator", "Desktop Technician/ scheduling manager", "Desktop support",
        "Deskside Support", "Deskside Technician", "Deskside support analyst"
    ],

    "IT Help Desk Support Specialist": [
        "IT SUPPORT ANALYST", "IT SUPPORT SPECIALIST I", "IT SUPPORT TECHNICIAN", "IT Support Analyst",
        "IT Support Analyst - Technology Services and Security", "IT Support Analyst II", "IT Support Analyst Intern",
        "IT Support Analyst Tier 2", "IT Support Analyst Tier III", "IT Support Analyst \\ IT Security Analyst",
        "IT Support Specialist", "IT Support Specialist / Network Administrator",
        "IT Support Specialist / Project Manager",
        "IT Support Specialist I", "IT Support Specialist II",
        "IT Support Specialist/ Assistant Database administrator",
        "IT Support Supervisor", "IT Support Supervisor / Project Manager", "IT Support Technician",
        "IT Support Technician II / IT Field Technician", "IT Support Technician III", "IT Support Technician/ Analyst",
        "IT Support specialist", "IT Support/ Security Analyst", "IT Support/Management", "IT Support/Networking"
    ],

    "Network Support": [
        "Network Support Administrator", "Network Support Engineer", "Network Support Manager",
        "Network Support Specialist", "Network Support Specialist Level Three & Network Administrator",
        "Network Support Technician", "Network Support Technician/ Network Administrator"
    ],

    "IT Technician": [
        "IT Technician", "IT Technician II", "IT Technician Level 2", "IT Technician and Project Manager",
        "IT assistant Computer Technician", "Technician", "Technology Support Specialist",
        "Tech Support", "Tech Support IV", "Tech Support Specialist", "Computer Technician",
        "Computer Technician team leader", "Computer Technician/ Network Administrator",
        "Computer and Network Technician", "Computer technician", "PC Technician",
        "PC Technician (Contract)", "PC Technician II", "PC/ Network Support Technician"
    ],

    "Enterprise Solutions & ERP": [
        "ERP Application Developer/EDI developer", "ERP Business Analyst", "ERP Consultant (PM)",
        "ERP Manager", "ERP Project Manager", "ERP Security Analyst", "ERP Systems Administrator",
        "SAP BUSINESS ONE TECHNICAL CONSULTANT", "SAP Database Administrator", "SAP Developer",
        "SAP Project Manager", "SAP SECURITY SPECIALIST", "SAP Security Administrator",
        "SAP Security Analyst", "SAP Security Consultant", "SAP Specialist",
        "Salesforce Admin and Developer", "Salesforce Admin/ Developer", "Salesforce Administrative Coordinator",
        "Salesforce Administrator", "Salesforce Administrator/ Developer", "Salesforce Administrator/Dev",
        "Salesforce Business Analyst", "Salesforce Consultant", "Salesforce Database Administrator/ Project Manager",
        "Salesforce Developer", "Salesforce Developer / Admin", "Salesforce Developer / Administrator",
        "Salesforce Developer/ Admin", "Salesforce Developer/Administrator", "Salesforce Developer/Lightning Developer"
    ],

    "Middleware & Integration Developer": [
        "Apache Camel Developer", "Appian BPM Consultant", "Appian Developer", "Integration Analyst",
        "Integration Developer", "Integration Engineer", "Mule ESB Consultant", "Mule ESB Developer",
        "MuleSoft Developer", "MuleSoft Developer / Mule ESB Developer", "MuleSoft Enterprise Architect (consultant)",
        "MuleSoft&Java Developer/Application Programmer", "Mulesoft Consultant", "Mulesoft Developer",
        "Mulesoft Integration Architect / Developer Lead", "Sr. Mule ESB Integration Developer",
        "Sr. MuleSoft Developer", "Sr. MuleSoft Solution Lead", "Sr. Mulesoft Developer",
        "Sr. Mulesoft Integration Developer"
    ],

    "Data Warehouse & BI Developer": [
        "BI Developer", "BI Developer / Database Developer", "BI Engineer and Database Administrator",
        "BI Enterprise Data Warehouse Architect", "BI PROJECT MANAGER", "Data Warehouse Eng",
        "Datawarehouse Architect/ Lead Datawarehouse Consultant", "Data Engineer",
        "Data Engineer - Financial Performance & Analytics",
        "Data Engineer / Informatica administrator / ETL Programmer and Analyst", "Data Engineer IaaS and PaaS",
        "Data Engineer/Infrastructure Engineer", "Data Engineering Fellow", "Big Data / Hadoop Lead",
        "Big Data / Spark Engineer", "Big Data Developer", "Big Data Engineer", "Big Data Project Lead",
        "Big Data Tech Lead / Architect", "Big data Developer", "BigData & Cloud Solutions Engineer",
        "Bigdata developer", "Hadoop", "Hadoop & Kafka Administrator", "Hadoop / Spark Developer",
        "Hadoop Admin", "Hadoop Administrator", "Hadoop Developer", "Hadoop developer",
        "Hadoop/Couchbase Administrator", "Hadoop/Spark Developer", "Hadoop/spark Developer"
    ],

    "Quality Assurance & Testing": [
        "QA ANALYST II", "QA Analyst", "QA Automation Engineer", "QA Automation Tester", "QA ENGINEER",
        "QA Engineer", "QA Engineer Software Test", "QA ITIL & Security Analyst", "QA Tester",
        "Quality Assurance Analyst", "Quality Assurance Analyst Co-op", "Quality Assurance Associate",
        "Quality Assurance Auditor", "Quality Assurance Coordinator", "Quality Assurance Evaluator",
        "Quality Assurance Inspector", "Quality Assurance Lead", "Quality Assurance Project Manager",
        "Quality Assurance Specialist", "Quality Auditor", "Quality Control", "Quality Control Analyst",
        "Quality Engineer", "Quality Engineer III", "Quality Operations Project Manager II",
        "Quality Specialist", "Quality Technician"
    ],

    "Systems/Network Operations": [
        "NOC Analyst", "NOC ENGINEER", "NOC Engineer", "NOC Engineer II", "NOC II ENGINEER",
        "NOC Network Specialist", "NOC SUPERVISOR / IT BUSINESS ANALYST", "NOC Technician",
        "NOC Technician / Help Desk Support / Network Administrator / Systems Administrator",
        "NOC Technician/ Network Engineer", "NOC specialist", "Network Operations Administrator",
        "Network Operations Center Analyst", "Network Operations Engineer", "Network Operations IT Professional",
        "Network Operations Manager", "Network Operations Officer", "Network Operations Support Engineer",
        "Network Operations Tech II", "Network Operations Technician", "Operations Support Analyst",
        "Operations Engineer", "Operations Engineer/ Network/Systems Admin/Systems Engineer",
        "Operations Manager", "Operations Manager of Information Technology", "Operations Support Analyst"
    ],

    "IT Security Compliance & Governance": [
        "Compliance Analyst", "Compliance Project Leader / Project Manager",
        "Compliance Risk Management - Governance lead",
        "Compliance Specialist/Learning Project Manager", "Compliance Underwriter (Contract)",
        "Compliance and Security Analyst", "IT Compliance & ERP Security Analyst", "IT Compliance & Security Analyst",
        "IT Compliance Analyst", "IT Compliance Manager", "Governance", "Governance Risk and Compliance Manager",
        "GRC Analyst", "IT SECURITY and Audit/ Compliance Analyst", "INFORMATION SECURITY MANAGER",
        "INFORMATION SECURITY ENGINEER", "Chief Information Security Officer (CISO)", "Chief Security Architect",
        "Deputy Chief Information Security Officer", "Director - IT Infrastructure",
        "Director - IT Service Management & Governance"
    ],

    "IT Audit & Risk Management": [
        "IT AUDITOR", "IT Audit Analyst", "IT Audit Consultant", "IT Audit Intern", "IT Audit Manager",
        "IT Audit Senior", "IT Audit Support Specialist", "IT Auditor", "IT Auditor -Contract/Temporary",
        "IT Auditor /Cyber Security Analyst", "IT Auditor/ IT Security", "IT Auditor/ Information Security Analyst",
        "Internal Control Analyst", "Internal Auditor", "IT Risk Analyst", "IT Risk Analyst & Auditor",
        "IT Risk Assurance & Compliance Analyst", "IT Risk Management Specialist", "IT Risk Manager",
        "IT Risk Security Compliance Analyst", "IT Risk and Compliance Analyst II",
        "IT Risk and Security Project Manager"
    ],

    "Infrastructure & Operations Manager": [
        "IT Infrastructure Analyst", "IT Infrastructure Associate Project Manager", "IT Infrastructure Engineer",
        "IT Infrastructure Engineer - Global IT Operations", "IT Infrastructure Engineering Security Lead",
        "IT Infrastructure Lead", "IT Infrastructure Manager", "IT Infrastructure Manager\\Service Delivery Manager",
        "IT Infrastructure Program / Project Manager", "IT Infrastructure Program Manager",
        "IT Infrastructure Project Manager", "IT Infrastructure Project Manager - Global Technology",
        "IT Infrastructure Security Analyst", "IT Infrastructure Specialist", "IT Infrastructure Transition Manager",
        "IT Infrastructure and Service Delivery Manager", "Director IT Infrastructure", "Director of IT Infrastructure",
        "Director of IT Infrastructure and Operations", "Director of IT Operations"
    ],

    "IT Director & CIO": [
        "DIRECTOR OF INFORMATION TECHNOLOGY", "DIRECTOR OF INFORMATION SECURITY", "DIRECTOR OF SECURITY",
        "DIRECTOR", "DIRECTOR - Shared Technologies", "DIRECTOR BUSINESS INTELLIGENCE",
        "DIRECTOR OF INFORMATION SECURITY",
        "Chief Information Officer", "Chief Information Officer (CIO)", "Chief Technology Officer",
        "Chief Technology Officer (CTO)",
        "Chief Techniology Officer / Co-Founder", "CIO & IT Network Administrator", "CIO COE Team Project Manager",
        "Director", "Director & Senior Director", "Director - IT Infrastructure", "Director IT",
        "Director IT Cloud and Application Support", "Director IT Infrastructure (Technical Services & Operations)",
        "Director IT Infrastructure and Security", "Director IT Project Management",
        "Director IT/Enterprise Services Program Manager"
    ],

    "Document Management & Content": [
        "CMS Senior Engineer", "CMS WordPress Administrator / Web Developer 2; Consulting",
        "HTML Content Manager (Contract)", "Content Analyst", "Content Developer", "Content Manager",
        "Content Review Analyst", "Content Strategist", "Document Management", "SharePoint & K2 Administrator",
        "SharePoint Administrator", "SharePoint Administrator/SharePoint Developer", "SharePoint Analyst",
        "SharePoint BI Tool Developer (SSIS", "SharePoint Developer", "SharePoint Developer/Web Designer",
        "SharePoint Project Specialist Consultant", "Sharepoint Architect", "Sharepoint Manager",
        "Sharepoint Support Analyst", "Web Content Coordinator", "Web Content Manager and Front End Developer",
        "Web Content Specialist", "Webmaster", "Website Administrator"
    ],

    "IT Asset & Vendor Management": [
        "IT Asset Analyst", "IT Asset Management and CMDB Product Manager", "IT Asset Manager",
        "Asset Management Audit", "Asset Protection Associate", "Asset Protection Specialist",
        "Buyer", "Buyer Contract Analyst", "Buyer Planner", "IT Procurement Analyst & Assistant Asset Manager",
        "IT Procurement Manager / Project Manager", "IT Vendor Management Specialist",
        "IT Vendor Relationship Analyst", "Procurement Coordinator", "Procurement Professional",
        "Vendor Management Specialist", "Vendor Risk Management Consultant", "Vendor Security Analyst"
    ],

    "IT Service Management": [
        "IT Service Analyst", "IT Service Desk Agent - Level 1", "IT Service Desk Analyst",
        "IT Service Desk Analyst II",
        "IT Service Desk Coordinator", "IT Service Desk Specialist", "IT Service Desk Support Engineer",
        "IT Service Desk Technician", "IT Service Management Consultant", "IT Service Manager",
        "IT Service Project Manager", "IT Service Specialist", "IT Service Supervisor", "IT Services Manager",
        "IT Services Specialist", "Service Desk Administrator / System Administrator / Security Analyst",
        "Service Desk Analyst", "Service Manager/ IT Change Manager", "ServiceNow Admin/ Developer",
        "ServiceNow Administrator/Developer", "ServiceNow Developer"
    ],

    "Telecommunications & VOIP": [
        "Cisco UC Phone Support", "Telecommunications", "Telecommunications Administrator",
        "Telecommunications Engineer", "Telecommunications Engineer III", "Telecommunications Field Engineer",
        "Telecommunications Manager / Sr. Network Administrator", "Telecommunications Technician",
        "Telecom Analyst", "Telecom Engineer", "Telecom Project Manager", "VOICE NETWORK ADMINISTRATOR",
        "VOICE SOLUTIONS ENGINEER", "VOIP/ network security engineer", "Unified Communications Administrator",
        "Unified Communications Engineer"
    ],

    "GIS & Spatial Analysis": [
        "GIS Analyst/ IT", "GIS Database Administrator", "GIS Technician", "GIS/ Database Administrator",
        "Geospatial Database Administrator/ Programmer"
    ],

    "Education & Training": [
        "CODING INSTRUCTOR", "TEACHING ASSISTANT", "Adjunct Faculty", "Adjunct Instructor",
        "Adjunct Instructor in Cybersecurity", "Adjunct Professor", "Adjunct Remote Professor",
        "Advanced Leadership Senior Instructor", "Affiliate Faculty - Computer Programming",
        "Assistant Professor", "Computer Science Learning Center Coach", "Computer Science teacher",
        "Creative Coding Instructor", "Faculty/ Database Administrator", "Instructor",
        "Instructor - Cybersecurity", "Instructor Mathematics", "Instructional Assistant",
        "Instructional Designer", "Instructional Project Designer", "Instructional Technology Consultant/Help Desk",
        "Learning And Development Specialist", "Professor", "Professor/Program Chair", "Teaching Assistant",
        "Teaching Assistant (TA)", "Udemy Instructor"
    ],

    "Healthcare IT": [
        "Clinical Database Builder", "Clinical Database Designer II", "Clinical IT Analyst",
        "Clinical IT Product Owner", "Clinical LIMS Analyst (Database Administrator)", "Clinical Systems Administrator",
        "Ambulatory Epic Analyst/Print Management", "Epic Ambulatory Analyst", "Epic Consultant",
        "Epic Desktop Support", "Epic Security Analyst", "Epic Support Technician",
        "Epic Technical Consultant/I.T. Project Manager",
        "EHR System Administrator", "EMR Help Desk", "Health IT Consultant", "Health Information Manager",
        "Health Information Technician/Privacy and Security Technician",
        "Health Information Technology PC Technician (Contractor)",
        "Healthcare Data Analyst", "Healthcare IT Project Manager"
    ],

    "Finance & Accounting IT": [
        "Account & HR Admin", "Account Administrator", "Account Clerk", "Accounting Clerk",
        "Accounting Manager/Assistant Controller", "Accounting Paraprofessional / Director of IT",
        "Accounting Specialist", "Accounting Specialist II",
        "Accounting Supervisor & Assistant to Business Manager/Accounts Payable & Purchasing",
        "Accounts Payable Accountant", "Accounts Payable Manager", "Accounts Payable Specialist",
        "Accounts Receivable Manager/Build-It-Back", "Budget & Management Analyst", "Budget & Planning Analyst",
        "Budget Analyst", "Finance Operations Admin - Lead Database Management", "Finance Project Manager",
        "Finance Reporting Analysis", "Financial Analyst", "Financial Analyst/HR Administrator",
        "Financial Management Analyst / Assistant Security Manager", "Financial Representative",
        "Financial Resolution Specialist", "Financial Security Analyst", "Financial analysis and business strategist"
    ],

    "Consulting & Professional Services": [
        "CONSULTANT", "CONSULTANT AND FREELANCER FRONT END DEV", "CONSULTANT/ FRONT- END DEVELOPER",
        "Advisory Consultant", "Advisory Project Specialist/Sr. Consultant", "Advisory Software Engineer",
        "Consultant", "Consultant (web developer)", "Consultant - Cyber and Information Security Recruiter",
        "Consultant - IT Project Manager", "Consultant - Supply Chain Management",
        "Consultant - Technical Implementation Specialist (Remote)",
        "Consultant - Technical Project Manager", "Consultant Project Management", "Consultant Security and Complance",
        "Consultant Senior Project Manager / Release Manager", "Consultant in UX/UI Design / Web Development",
        "Consultant | Cyber Security", "Consultant/ Project Manager", "Consultant/Contractor", "Consultant/Owner",
        "Consulting Engineer", "Consulting Project Manager / IT / Procces Control", "Consulting Security Specialist",
        "Consulting Senior Analyst", "Consulting Senior Database Developer", "Consulting Systems Engineer",
        "Consulting Systems Engineer I (SOC Lead)", "Consulting Technical Project Manager",
        "Consutant\\Senior Network administration Consutant\\Senior Network administration"
    ],

    "Freelance & Independent Contractor": [
        "FREELANCE", "FREELANCE CONSULTANT WEBISTE DEVELOPMENT", "FREELANCE DESIGNER / FRONT END DEVELOPER",
        "FREELANCE Email Marketing Campaign Administration", "FREELANCE FRONT- END DEVELOPER / DIGITAL CONSULTANT",
        "FREELANCE FRONT- END WEB DEVELOPER", "FREELANCE FRONT- END WEB DEVELOPER AND FREELANCE WRITER",
        "FREELANCE FRONT- END DEVELOPER / UX DESIGNER", "FREELANCE FRONT- END WEB DEVELOPER",
        "FREELANCE FRONT-END DEVELOPER", "FREELANCE FRONT-END DEVELOPER INTERN", "Freelance",
        "Freelance - Front-end Developer/Designer", "Freelance - Mathematics Tutor", "Freelance / Front-end developer",
        "Freelance Android Developer", "Freelance Consultant and Developer", "Freelance Corporate Trainer",
        "Freelance Creative Director / Designer", "Freelance Creative Technologist", "Freelance Database Administrator",
        "Freelance Designer", "Freelance Designer & Consultant", "Freelance Developer", "Freelancer",
        "Freelancer Front- End Developer/Designer", "Freelancing", "Independent Consultant",
        "Independent Consultant & Senior Security Advisor",
        "Independent Consulting", "Independent Contractor", "Independent Freelance IT Consultant",
        "Independent IT Consultant", "Self-Employed"
    ],

    "HR & Administration": [
        "ADMINISTRATIVE ASSISTANT", "ADMINISTRATIVE SUPPORT SPECIALIST (MARKETING & TRAINING)",
        "Administrative Assistant", "Administrative Assistant (Temp)", "Administrative Assistant/ Front End Developer",
        "Administrative Assistant/ Project Manager", "Administrative Assistant/Receptionist",
        "Administrative Assistant/Safety Coordinator",
        "Administrative Coordinator", "Administrative Director", "Administrative Intern - Office",
        "Administrative Office Manager (Contract position)", "Administrative Officer",
        "Administrative Project Analyst/ IT Helpdesk Assistant (Intern)",
        "Administrative Secretary", "Administrative Services", "Administrative and Database Assistant",
        "Administrative assistant II",
        "Administrator", "Administrator &Gym Manager", "Administrator ( Infrastructure/Systems Administration T3)",
        "Administrator and Family Coordinator", "Administrator(s)", "Executive Assistant",
        "Executive Assistant & Database/Market Administrator",
        "Executive Assistant and Database Manager", "Executive Assistant to CEO",
        "HR Assistant / Database Administrator",
        "HR Project Administrator", "HR/Logistics Manager", "HRIS Analyst I", "Human Resources Manager",
        "Office Administrator", "Office Assistant", "Office Coordinator/Scheduler", "Office Manager",
        "Office Manager & Paralegal", "Office Manager & Sales Representative", "Office Manager/Paralegal",
        "Office Manager/Partner",
        "Office Receptionist/Vet Assistant", "Office Services Assistant", "Office Staff/ Database Administrator",
        "Office and Project Manager", "Receptionist", "Receptionist / Special Project Assistant"
    ],

    "Facilities & Operations": [
        "FORKLIFT OPERATOR/MATERIAL HANDLER", "Building Shift Manager", "Building and Facilities Services Manager",
        "Facilities Technician", "Facilities and Project Manager", "Facilities and Project Manager",
        "Facilities Operations Manager", "Grounds Management", "Operations", "Facilities Manager"
    ],

    "Marketing & Communications": [
        "DIGITAL MARKETING MANAGER", "DIGITAL MARKETING STRATEGIST", "DIGITAL MKTG MANAGER",
        "Digital Marketing Analyst", "Digital Marketing Consultant", "Digital Marketing Director",
        "Digital Marketing Lead", "Digital Marketing Manager", "Digital Marketing Specialist",
        "Email Marketing Specialist", "Email Specialist/ Front- End Developer", "Marketing Analyst",
        "Marketing Consultant", "Marketing Content Writer", "Marketing Director", "Marketing Intern",
        "Marketing Manager", "Marketing Operations Manager", "Marketing Project Manager",
        "Marketing Specialist", "Marketing Web Developer", "Marketing and Compliance Project Manager",
        "Marketing and IT Coordinator", "Marketing and Sales Manager", "Social Media Consultant",
        "Social Media Coordinator", "Social Media Manager"
    ],

    "Sales & Account Management": [
        "Account Executive", "Account Executive/ Project Manager", "Account Management Specialist",
        "Account Manager", "Account Manager / Mobile Apps", "Account Manager IT Services",
        "Account Manager/Logistics Coordinator",
        "Account Service Manager", "Account Services Assistant", "Account Support Manager/ IT Project Manager",
        "Account Payable/Receivable Specialist", "Business Account Manager", "Corporate Account Manager",
        "National Account Coordinator", "Sales Manager", "Sales Representative", "Sales Specialist",
        "Strategic Account Manager"
    ],

    "Logistics & Supply Chain": [
        "SUPPLY CONTROL & PRODUCTION ANALYST", "Acquisition Program Manager", "Cargo Handler",
        "Commercial Parts Driver", "Commercial Driver", "Commercial Driver - Shuttle Operator",
        "Courier", "Delivery Driver", "Dispatcher", "Distribution Operator", "Driver",
        "Driver & Mentor", "Driver Partner", "Driver Supervisor", "Driver/Admin", "Driver/Contractor",
        "Driver/Operator", "Freight Forwarder/ Warehouse System Admin",
        "Logistics & Transport Officer/ Procument Officer",
        "Logistics Analyst", "Logistics Developer", "Logistics Operations Manager", "Logistics Project Manager",
        "Logistics Sales Manager", "Logistics/Dispatcher", "Logistics/Support (Part Time)", "Material Handler",
        "Materials Planner", "Order Management", "Order Selector", "Procurement Coordinator",
        "Receiving Associate", "Shipping & Receiving Clerk", "Warehouse Assistant", "Warehouse Database Administrator",
        "Warehouse Manager"
    ],

    "Legal & Compliance": [
        "Compliance Officer", "Compliance Officer", "Contracts Administrator (Aircraft Interiors)",
        "Contracts Manager", "Paralegal", "Litigation Paralegal", "Legal Analyst", "Legal Assistant",
        "Legal Operations Associate"
    ],

    "Hospitality & Front Desk Manager": [
        "Front Desk Administrator", "Front Desk Manager", "Front Desk Receptionist", "Store Manager",
        "Store Supervisor",
        "Shift Leader"
    ],

    "Restaurant & Culinary Staff": [
        "Barista", "Barista/Associate", "Barista/Food Prep", "Bartender", "Bartender/Barback/Support",
        "Bartender/Room Service", "Busser/Server", "Cashier/cook", "Cook", "Grill Operator/ Server",
        "Kitchen help", "Kitchen staff / Prep / Dishwashing / Delivery Meal Driver", "Line Cook", "Server",
        "Server / Busser", "Server/Bartender", "Server/Waiter", "Dishwasher"
    ],

    "Retail & Merchandise Operations": [
        "Cashier", "Cashier / Stock Person", "Cashier/Customer Service", "Cashier/Sales Associate/Customer Service",
        "Merchandise Coordinator", "Stocker/Receiver", "Store Associate", "Stock Associate"
    ],

    "Warehouse & Fulfillment Operations": [
        "Amazon warehouse associate", "Crew Member", "Package Handler", "Warehouse Worker",
        "Warehouse Sortation Associate/FC Associate I"
    ],

    "Customer Service Representative": [
        "Customer Service", "Customer Service Administrator", "Customer Service Associate",
        "Customer Service Representative", "Customer Service Specialist", "Customer Service Supervisor",
        "Customer Support Agent", "Customer Support Representative & Technical Support", "Customer Support Supervisor"
    ],

    "Healthcare Support Staff": [
        "Housekeeper", "Orderly", "Part-Time Office Administrator"
    ],

    "Sales Support & Retail": [
        "Account Clerk", "After-Sales Representative", "Beauty Advisor/Sales Associate", "Camera Sales Professional",
        "Counter Sales", "Home Design and Construction Specialist", "Home Furnishings Sales Associate",
        "Inside Sales Representative", "Inventory Control Specialist", "Inventory Database Coordinator",
        "JCPenney Sales Associate", "Panera Bread Associate", "Rental Sales Agent", "Retail Sales Associate",
        "Retail Sales Supervisor", "Sales Associate", "Sales Associate - Furniture", "Sales Associate/ Part-time",
        "Sales Consultant (part-time)", "Sales Coordinator /Sales Engineer", "Sales Development Representative",
        "Sales Engineer", "Sales Engineering Manager", "Sales Intern", "Sales Leader/ Manager (part-time)"
    ],

    "Legal & Compliance Professional": [
        "Advocate", "Associate Attorney", "Associate Counsel"
    ],

    "Business Advisor & Consultant": [
        "Advisor", "Associate Brand Manager", "Associate Consultant", "Associate Consultant (Remote)"
    ],

    "Apprenticeship Program": [
        "Apprentice", "Apprentice - IT Project Manager", "Apprentice HVAC", "Apprentice Personnel",
        "Apprentice UX/UI Designer and Front-End Developer"
    ],

    "Creative & Artistic Professional": [
        "Artist", "Professional Musician- Guitar Player"
    ],

    "Associate-Level IT & Technical Specialist": [
        "Assoc security analyst", "Associate Cyber Security Analyst", "Associate Data Scientist",
        "Associate Professional Database Administrator", "Associate Professional Database Engineer - Operations Team",
        "Associate Security Analyst", "Associate Security Consultant", "Associate Security Engineer",
        "Associate Software Developer", "Associate Software Engineer", "Associate System Administrator",
        "Associate System Administrator / Network Engineer", "Associate Systems Developer",
        "Associate Systems Engineer", "Associate Technical Consultant", "Associate Technical Lead",
        "Associate Technical consultant", "Associate Technology Risk Analyst"
    ],

    "Associate & Manager Level Executive": [
        "Assoc. Director", "Associate Director", "Associate Director - IT Security", "Associate Manager",
        "Associate Manager II", "Associate Program Director", "Associate Project Manager",
        "Associate Project Manager IT", "Associate Project manager", "Associate manager"
    ],

    "Software Architect": [
        "Software Architect", "Solutions Architect", "Solutions Architect / Senior Software Engineer",
        "Solutions Architect AWS", "Solutions Architect for IoT", "Solutions Architect/ IT Consultant",
        "Enterprise Architect", "Enterprise Solutions Architect", "Application Architect",
        "Integration Architect", "Security Architect", "Infrastructure Architect",
        "Chief Architect", "Lead Software Architect", "Principal Architect",
        "Principal Software Architect", "Senior Solutions Architect", "Senior Systems Architect"
    ],

    "Web Developer": [
        "Web Developer", "Web Developer Intern", "Web Developer II", "Web Developer III",
        "Web Developer (Contract)", "Web Developer (PHP/MySQL)", "Web Developer - Part Time",
        "Web Developer / Designer", "Web Developer / Software Developer", "Web Developer and Graphic Designer",
        "Web Developer Internship", "Web Developers", "Sr. Web Developer", "Senior Web Developer",
        "Web Developer & Project Manager", "Website Developer", "Website Developer (Freelance)",
        "Web Developer/Designer", "Remote Web Developer", "Full-Time Web Developer", "Part-Time Web Developer"
    ],

    "WordPress Developer": [
        "WordPress", "WordPress & WooCommerce Specialist", "WordPress Administrator",
        "WordPress Consultant", "WordPress Developer", "WordPress Developer & Designer",
        "WordPress Designer", "WordPress Developer/Trainer", "WordPress Specialist",
        "WordPress Web Developer", "Senior WordPress Developer", "Wordpress Developer",
        "WP/PHP Developer", "WordPress and PHP Developer"
    ],

    "PHP & Backend Web": [
        "PHP Developer", "PHP Developer / Full Stack Developer", "PHP Full Stack Developer",
        "PHP Programmer", "PHP Web Developer", "Sr. PHP Developer", "Senior PHP Developer",
        "PHP & JavaScript Developer", "PHP/MySQL Developer", "LAMP Stack Developer",
        "Zend Framework Developer", "Laravel Developer"
    ],

    "Ruby & Rails Developer": [
        "Ruby Developer", "Ruby on Rails Developer", "Ruby Rails Developer",
        "Rails Developer", "Senior Rails Developer", "Sr. Ruby Developer",
        "Ruby Programmer", "Ruby Full Stack Developer"
    ],

    "C/C++ Developer": [
        "C Developer", "C++ Developer", "C++ Programmer", "C/C++ Developer",
        "C/C++ Developer/ Cloud Engineer", "C/C++ Embedded Systems Developer",
        "Objective-C Developer", "C# Developer", "Senior C++ Developer",
        "Sr. C++ Developer"
    ],

    "Go/Golang Developer": [
        "Golang Developer", "Go Developer", "Go/Golang Developer",
        "Backend Python developer and occasional Go developer"
    ],

    "Rust Developer": [
        "Rust Developer", "Rust Systems Developer"
    ],

    "Scripting & Automation": [
        "Scripting Engineer", "Shell Script Developer", "Bash Developer",
        "PowerShell Administrator", "Automation Engineer", "Test Automation Developer",
        "Automation Specialist", "Automation QA Engineer", "RPA Developer"
    ],

    "x86-64 Assembly": [
        "x86-64 Assembly Developer"
    ],

    "ETL & Data Integration": [
        "ETL", "ETL Analyst", "ETL Developer", "ETL Developer / BI",
        "ETL Developer / BI Developer", "ETL Developer / Data Warehouse Developer",
        "ETL Developer and SQL Developer", "ETL Developer/ Informatica administrator / ETL Programmer and Analyst",
        "ETL Engineer", "ETL Informatica Developer", "ETL Lead", "ETL Specialist",
        "ETL Developer / Analyst", "Informatica Developer"
    ],

    "Performance Testing": [
        "Performance Test Engineer", "Performance Tester", "Performance Testing Engineer",
        "Load Test Engineer", "Load Testing Specialist"
    ],

    "Security Testing": [
        "Penetration Tester", "Penetration Testing Specialist", "Security Tester",
        "Vulnerability Assessment Specialist"
    ],

    "Release Management": [
        "Release Engineer", "Release Manager", "Release Coordinator",
        "Release Management Specialist"
    ],

    "Configuration Management": [
        "CM Engineer", "Change & Configuration Manager", "Configuration Manager",
        "Change Manager", "Change Control Manager"
    ],

    "Information Architecture": [
        "Information Architect", "IA/UX Designer", "Information Architecture Designer"
    ],

    "Systems Engineer": [
        "Systems Engineer", "Systems Engineer I", "Systems Engineer II",
        "Systems Engineer III", "Systems Engineer IV", "Senior Systems Engineer",
        "Sr. Systems Engineer", "Sr Systems Engineer", "Enterprise Systems Engineer",
        "Systems Engineering Manager", "Lead Systems Engineer",
        "Systems Engineer/Architect"
    ],

    "Solutions Engineer": [
        "Solutions Engineer", "Solutions Engineer I", "Solutions Engineer II",
        "Solutions Engineer III", "Solutions Engineer/Associate", "Sales Solutions Engineer",
        "Pre-Sales Solutions Engineer", "Senior Solutions Engineer",
        "Sr. Solutions Engineer"
    ],

    "Software Engineer": [
        "Software Engineer", "Software Engineer I", "Software Engineer II",
        "Software Engineer III", "Software Engineer IV", "Software Engineer V",
        "Software Engineer Intern", "Software Engineer Internship",
        "Software Engineer/ Database Developer", "Software Engineer (Contract)",
        "Software Engineer (Full Stack)", "Software Engineer - Mobile",
        "Software Engineer - Solutions", "Software Engineer and Artist",
        "Senior Software Engineer", "Sr. Software Engineer", "Sr Software Engineer",
        "Principal Software Engineer", "Lead Software Engineer",
        "Staff Software Engineer", "Advisory Software Engineer"
    ],

    "Application Developer": [
        "Application Developer", "Application Developer I", "Application Developer II",
        "Application Developer III", "Application Developer & UX Designer",
        "Applications Developer", "Application Support Developer",
        "Automotive Applications Developer", "Senior Application Developer",
        "Sr. Application Developer", "Sr Application Developer",
        "Sr. Application Software Developer", "Applications Programmer"
    ],

    "Object-Oriented Specialist": [
        "Object-Oriented Developer / Lead", "OOP Developer"
    ],

    "MVC Framework Developer": [
        "MVC Developer", "ASP.NET MVC Developer", "Spring MVC Developer"
    ],

    "API & Web Services Developer": [
        "API Developer", "API Engineer", "REST API Developer",
        "Web Services Developer", "Web Services Engineer",
        "Microservices Developer", "Microservices Architect",
        "API & Microservices Developer"
    ],

    "IoT & Embedded Systems": [
        "IoT Developer", "IoT Engineer", "Embedded Developer",
        "Embedded Software Developer", "Embedded Systems Developer",
        "Embedded Software Engineer", "Firmware Developer", "Firmware Engineer",
        "Hardware Engineer", "Hardware Developer"
    ],

    "Blockchain Developer": [
        "Blockchain Developer", "Blockchain Engineer", "Smart Contract Developer",
        "Crypto Developer", "Web3 Developer"
    ],

    "AI & Machine Learning Engineer": [
        "AI", "AI Developer", "AI Engineer", "AI Research Scientist",
        "Machine Learning Developer", "Machine Learning Engineer",
        "ML Engineer", "ML Specialist", "AI/Machine Learning Developer",
        "Deep Learning Engineer", "NLP Engineer", "Computer Vision Engineer",
        "Senior Machine Learning Engineer"
    ],

    "DevOps & Infrastructure": [
        "DevOps", "DevOps (Front End Developer)", "DevOps Cloud Architect",
        "DevOps DBA", "DevOps Engineer", "DevOps Engineer - (Technical Consultant)",
        "DevOps Engineer/AWS", "DevOps Lead", "DevOps Manager",
        "DevOps/Middleware Developer and System Integrations", "Devops Engineer",
        "Senior DevOps Engineer", "Sr. DevOps Engineer", "Sr. DevOps/Cloud Engineer"
    ],

    "Build & Release Management": [
        "Build Manager", "Build Engineer", "Maven Developer",
        "Gradle Developer"
    ],

    "Geospatial & Mapping": [
        "Geospatial Analyst", "Geospatial Developer", "GIS Specialist",
        "Mapping Specialist", "Spatial Analyst"
    ],

    "CRM Specialist": [
        "CRM Administrator", "CRM Developer", "CRM Business Analyst",
        "CRM Manager", "Dynamics CRM", "Dynamics CRM Developer",
        "Salesforce CRM Developer"
    ],

    "Project Control & Analytics": [
        "Project Controls", "Project Management Control", "Project Analyst",
        "PMO Analyst"
    ],

    "Operations Specialist": [
        "Operations Analyst", "Operations Assistant", "Operations Coordinator",
        "Operations Specialist", "Ops Manager"
    ],

    "Security Infrastructure": [
        "Certificate Authority Administrator", "PKI Administrator",
        "VPN Administrator", "Firewall Administrator", "DMZ Administrator",
        "Proxy Administrator", "WAF Administrator"
    ],

    "Backup & Disaster Recovery": [
        "Backup Administrator", "Backup Specialist", "Disaster Recovery Administrator",
        "DR Specialist", "Business Continuity Specialist"
    ],

    "Virtualization Administrator": [
        "VMware Administrator", "VMware Engineer", "Hyper-V Administrator",
        "Virtual Machine Administrator", "Virtualization Specialist",
        "Virtualization Engineer", "Cloud Virtualization Administrator"
    ],

    "Storage & SAN Administrator": [
        "Storage Administrator", "Storage Engineer", "SAN Administrator",
        "SAN Engineer", "Storage Specialist", "Backup Storage Engineer"
    ],

    "Telecom & Voice Engineer": [
        "Telecom Analyst", "Telecom Engineer", "Telecom Project Manager",
        "Telecom Specialist", "VoIP Engineer"
    ],

    "Network Security": [
        "Network Security Analyst", "Network Security Engineer", "Network Security Administrator",
        "Network Security Manager"
    ],

    "Compliance & Audit Officer": [
        "Compliance Officer", "Audit Manager", "Audit Supervisor",
        "Auditor", "Internal Auditor", "External Auditor"
    ],

    "Quality Assurance Manager": [
        "QA Manager", "Quality Manager", "Quality Director"
    ],

    "Technical Writer": [
        "Technical Writer", "Technical Documentation Specialist",
        "Documentation Writer", "Technical Communications Specialist"
    ],

    "Solutions Consultant": [
        "Solutions Consultant", "IT Solutions Consultant", "Senior Solutions Consultant",
        "Solutions Consultant (Remote)"
    ],

    "Support Consultant": [
        "Support Consultant", "Technical Support Consultant", "Customer Support Consultant"
    ],

    "Implementation Specialist": [
        "Implementation Specialist", "Systems Implementation Specialist",
        "Implementation Manager", "Implementation Engineer",
        "Implementation Consultant"
    ],

    "Field Engineer": [
        "Field Engineer", "Field Support Engineer", "Field Service Engineer",
        "Field Technician", "Field Support Technician"
    ],

    "Helpdesk Manager": [
        "Helpdesk Manager", "Help Desk Manager", "Helpdesk Lead",
        "Helpdesk Supervisor"
    ],

    "Training & Development": [
        "Training Specialist", "Training Manager", "Training Coordinator",
        "Curriculum Developer", "Trainer", "Corporate Trainer",
        "Training Development Specialist"
    ],

    "IT Governance": [
        "IT Governance Manager", "IT Governance Specialist", "IT Governance Officer"
    ],

    "Enterprise Architecture": [
        "Enterprise Architect", "Enterprise Architecture Manager",
        "Enterprise Architecture Lead"
    ],

    "Service Level Manager": [
        "Service Level Manager", "SLA Manager", "Service Level Agreement Manager"
    ],

    "Technology Manager": [
        "Technology Manager", "Technology Director", "Technology Coordinator",
        "Technology Specialist"
    ],

    "Business Systems": [
        "Business Systems Analyst", "Business Systems Manager", "Business Systems Architect"
    ],

    "Quality Systems": [
        "Quality Systems Analyst", "Quality Systems Manager", "Quality Systems Engineer"
    ],

    "Facilities IT": [
        "Facilities IT Specialist", "Facilities Technology Manager"
    ]
}


# Step 1: Create reverse mapping (title → main category)
reverse_mapping = {}
for main_category, titles in job_role_mapping.items():
    for t in titles:
        reverse_mapping[t.lower()] = main_category

# Step 2: Normalize original categories (titles) and map them
df['Category'] = df['Category'].str.strip().str.lower().map(reverse_mapping)

# Step 3: Drop rows where mapping failed (Category = NaN)
df = df.dropna(subset=['Category'])

# Step 4: Drop categories with less than 50 samples
category_counts = df['Category'].value_counts()
valid_categories = category_counts[category_counts >= 190].index
print(valid_categories)

df = df[df['Category'].isin(valid_categories)]

# Verify
print(df['Category'].value_counts())

# Create list of tuples (category, count)
category_tuples = list(df['Category'].value_counts().items())

# Save cleaned dataset
df.to_csv("Final_Categorized.csv", index=False)

print(category_tuples)
Total=0
for i in category_tuples:
    Total+=i[1]
print(Total)


