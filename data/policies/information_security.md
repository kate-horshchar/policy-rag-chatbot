# Information Security Policy
**Sunshine Consulting**
*Effective Date: January 1, 2026 | Last Revised: March 30, 2026*
*Owner: IT Security Team | Approved by: Chief Information Security Officer (CISO)*

---

## 1. Purpose and Scope

This policy establishes the information security standards and responsibilities for all employees, contractors, and third-party vendors who access Sunshine Consulting systems, networks, or data. The goal is to protect the confidentiality, integrity, and availability of company and client information.

Non-compliance with this policy may result in disciplinary action up to and including termination.

---

## 2. Password Requirements

All accounts used to access Sunshine Consulting systems must comply with the following password standards:

- Minimum **12 characters** in length
- Must include at least: one uppercase letter, one lowercase letter, one number, and one special character
- Passwords must **not** contain the employee's name, username, or common words
- Passwords must be changed every **90 days**
- Reuse of the last **10 passwords** is prohibited
- Passwords must **never** be shared with anyone, including IT staff

### Password Manager
Employees are required to use the company-approved password manager **1Password** for storing and generating passwords. Contact the IT team via Slack **#it-help** to get access.

---

## 3. Multi-Factor Authentication (MFA)

Multi-factor authentication is **mandatory** for all employees on the following systems:

- Company email (Google Workspace)
- Rippling (HR platform)
- Internal project management tools (Notion, Jira)
- VPN access
- Any system containing Confidential or Restricted data

MFA must be set up using the **Google Authenticator** app or a hardware security key. SMS-based MFA is permitted only as a backup method.

---

## 4. Data Classification

All company and client data must be classified into one of the following categories:

| Classification | Description | Examples |
|---|---|---|
| **Public** | Information approved for public release | Marketing materials, job postings, public website content |
| **Internal** | General business information for internal use only | Internal announcements, org charts, meeting notes |
| **Confidential** | Sensitive business or client information | Client contracts, financial reports, employee salaries |
| **Restricted** | Highly sensitive data with strict access controls | Personal client data, passwords, security credentials, legal documents |

### Handling Rules by Classification

- **Public:** May be shared freely
- **Internal:** Do not share outside the company without manager approval
- **Confidential:** Share only with authorized personnel on a need-to-know basis; encrypt when sending via email
- **Restricted:** Access requires explicit CISO approval; must be encrypted at rest and in transit; never stored on personal devices

---

## 5. Device Policy

### 5.1 Company-Issued Devices
- Company laptops must have **full-disk encryption** enabled (BitLocker for Windows, FileVault for Mac)
- Automatic screen lock must be set to **5 minutes** of inactivity
- Only IT-approved software may be installed
- Devices must not be used by family members or third parties

### 5.2 BYOD (Bring Your Own Device)
Personal devices may be used **only** for the following:
- Reading company email via the approved mobile app (Google Workspace)
- Two-factor authentication
- Accessing the company intranet via VPN

Personal devices may **not** be used to:
- Download or store Confidential or Restricted data
- Access client systems or databases
- Install unauthorized software that interacts with company systems

All personal devices used for work purposes must have a screen lock enabled and must be reported to IT if lost or stolen.

---

## 6. VPN Usage

- Employees must connect to the Sunshine Consulting **VPN** whenever accessing internal systems remotely
- VPN is mandatory when working from public networks (cafes, airports, hotels)
- The approved VPN client is **Cisco AnyConnect**, available via the IT portal
- VPN credentials must not be shared under any circumstances

---

## 7. Client Data Handling

Sunshine Consulting handles sensitive data on behalf of clients. The following rules apply to all client data:

- Client data must be stored **only** in approved company systems (Google Drive, Notion, or client-specified platforms)
- Client data must never be stored on personal devices or unauthorized cloud storage (e.g., personal Dropbox, Google Drive personal accounts)
- Sharing client data with third parties requires written approval from the client and the CISO
- All client data must be deleted or returned at the end of the engagement as specified in the client contract

---

## 8. Incident Reporting

Any suspected or confirmed security incident must be reported **within 2 hours** of discovery. Security incidents include:

- Lost or stolen company or personal work devices
- Unauthorized access to systems or accounts
- Phishing emails clicked or credentials entered on suspicious sites
- Accidental sharing of Confidential or Restricted data
- Malware or ransomware detection

### How to Report
- **Slack:** #it-security (for immediate alerts)
- **Email:** security@sunshineconsulting.com
- **Phone:** IT Security Hotline — +1 (800) 555-0192 (available 24/7)

Failure to report a known security incident is a violation of this policy and may result in disciplinary action.

---

## 9. Prohibited Activities

The following activities are strictly prohibited on all company systems and networks:

- Installing unauthorized software or browser extensions
- Disabling antivirus, firewall, or security monitoring tools
- Accessing or attempting to access systems or data beyond your authorized scope
- Using company systems for cryptocurrency mining
- Sharing login credentials with colleagues, managers, or IT staff
- Storing Restricted data on unencrypted devices or unauthorized cloud services
- Circumventing security controls (e.g., using a personal hotspot to bypass VPN)

---

## 10. Security Awareness Training

All employees must complete the following training:

- **Annual security awareness training** — assigned via the IT portal each January
- **Phishing simulation tests** — conducted quarterly; employees who fail must complete additional training within 5 business days
- **New employee security onboarding** — completed within the first 5 days of employment

---

## 11. Contact

For questions or to report a security incident, contact the **IT Security team** via:
- Email: **security@sunshineconsulting.com**
- Slack: **#it-security**
- Emergency Hotline: **+1 (800) 555-0192** (24/7)
- For general IT help: Slack **#it-help** or **it@sunshineconsulting.com**
