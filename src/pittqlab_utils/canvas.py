"""
Low level wrapper around requests targeting the Canvas API
"""

import os
from typing import Any, Dict, List, Optional

import requests


class CanvasClient:
    def __init__(self, token: str, course_id: str):
        self.token = token
        self.course_id = course_id
        self.base_url = f"https://canvas.pitt.edu/api/v1/courses/{course_id}/"
        self.header = {"Authorization": f"Bearer {token}"}  # API key

    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        r = requests.get(self.base_url + url, headers=self.header, json=params)
        r.raise_for_status()
        data = r.json()

        # check for pagination and append as needed
        while "next" in r.links:
            r = requests.get(r.links["next"]["url"], headers=self.header)
            data += r.json()
        return data  # list of dicts

    def post(self, url: str, params: Dict[str, Any]) -> None:
        r = requests.post(self.base_url + url, headers=self.header, json=params)
        r.raise_for_status()

    def put(self, url: str, params: Dict[str, Any]) -> None:
        r = requests.put(self.base_url + url, headers=self.header, json=params)
        r.raise_for_status()

    def upload(self, url: str, file: str) -> Dict[str, Any]:
        # step 1
        params = {"name": os.path.basename(file), "size": os.path.getsize(file)}
        r = requests.post(self.base_url + url, headers=self.header, json=params)
        r.raise_for_status()
        upload_url = r.json()["upload_url"]
        upload_params = r.json()["upload_params"]

        # step 2
        with open(file, "rb") as f:
            r = requests.post(upload_url, json=upload_params, files={file: f})
        r.raise_for_status()
        location = r.headers["Location"]

        # step 3
        r = requests.get(location, headers=self.header)
        r.raise_for_status()
        return r.json()

    def download_attachment(self, url: str, file_path: str) -> str:
        """Download an attachment once; skip if already present."""
        if os.path.exists(file_path):
            return file_path

        r = requests.get(url, headers=self.header)
        r.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(r.content)
        return file_path

    # Helper functions for common operations

    def get_groups(self) -> Any:
        """Get all groups in the course."""
        return self.get("groups")

    def get_students(self) -> Any:
        """Get all students enrolled in the course."""
        return self.get("students")

    def get_assignments(self) -> Any:
        """Get all assignments in the course."""
        return self.get("assignments")

    def get_assignment(self, assignment_id: int) -> Any:
        """Get a specific assignment by ID."""
        return self.get(f"assignments/{assignment_id}")

    def get_submissions(self, assignment_id: int) -> Any:
        """Get all submissions for an assignment."""
        return self.get(f"assignments/{assignment_id}/submissions")

    def get_group_users(self, group_id: int) -> List[Dict[str, Any]]:
        """Get all users in a specific group."""
        # Groups endpoint is at root level, not course level
        r = requests.get(
            f"https://canvas.pitt.edu/api/v1/groups/{group_id}/users",
            headers=self.header,
        )
        r.raise_for_status()
        data = r.json()

        # Handle pagination
        while "next" in r.links:
            r = requests.get(r.links["next"]["url"], headers=self.header)
            data += r.json()
        return data

    def assign_peer_review(self, assignment_id: int, submission_id: int, reviewer_id: int) -> Dict[str, Any]:
        """
        Assign a peer review.

        Parameters:
        -----------
        assignment_id : int
            The assignment ID
        submission_id : int
            The submission ID (the work being reviewed)
        reviewer_id : int
            The user ID of the reviewer

        Returns:
        --------
        dict
            The peer review assignment response
        """
        url = f"assignments/{assignment_id}/submissions/{submission_id}/peer_reviews"
        params = {"user_id": reviewer_id}
        r = requests.post(self.base_url + url, headers=self.header, json=params)
        r.raise_for_status()
        return r.json()

    def get_attendance(self, attempt_lti_endpoint: bool = False) -> Optional[Any]:
        """
        Attempt to get attendance data.

        Note: Canvas doesn't have built-in attendance API. This function tries
        common endpoints that might exist if your institution has installed
        an attendance tool (like Roll Call Attendance).

        Parameters:
        -----------
        attempt_lti_endpoint : bool
            Whether to try LTI-based attendance endpoints

        Returns:
        --------
        dict or None
            Attendance data if available, None otherwise
        """
        # Try common attendance endpoints (may not exist)
        endpoints_to_try = [
            "attendance",
            "roll_call",
            "roll_call_attendance",
        ]

        for endpoint in endpoints_to_try:
            try:
                return self.get(endpoint)
            except Exception:
                continue

        # If LTI endpoint attempt is requested
        if attempt_lti_endpoint:
            try:
                # Some attendance tools use LTI endpoints
                r = requests.get(
                    f"https://canvas.pitt.edu/api/v1/courses/{self.course_id}/external_tools",
                    headers=self.header,
                )
                r.raise_for_status()
                tools = r.json()
                # Look for attendance-related tools
                for tool in tools:
                    if "attendance" in tool.get("name", "").lower() or "roll" in tool.get("name", "").lower():
                        print(f"Found attendance tool: {tool.get('name')}")
                        print(f"  Tool ID: {tool.get('id')}")
                        return tool
            except Exception:
                pass

        return None

    def get_student_activity(self, user_id: Optional[int] = None, limit_assignments: int = 5) -> Dict[str, Any]:
        """
        Get student activity/participation data as a proxy for attendance.

        This includes:
        - Assignment submissions
        - Quiz attempts
        - Discussion participation
        - Page views (if available)

        Parameters:
        -----------
        user_id : int, optional
            Specific user ID. If None, returns activity for all students.
        limit_assignments : int, optional
            Limit number of assignments to check (default: 5) to avoid too many API calls

        Returns:
        --------
        dict
            Activity data
        """
        activity_data: Dict[str, Any] = {}

        if user_id:
            # Get activity for specific user
            try:
                assignments = self.get_assignments()
                for assignment in assignments[:limit_assignments]:
                    try:
                        submissions = self.get_submissions(assignment["id"])
                        user_sub = [s for s in submissions if s.get("user_id") == user_id]
                        if user_sub:
                            activity_data[assignment["name"]] = {
                                "submitted": user_sub[0].get("submitted_at"),
                                "status": user_sub[0].get("workflow_state"),
                            }
                    except Exception:
                        continue
            except Exception as e:
                print(f"Error getting activity for user {user_id}: {e}")
        else:
            # Get activity summary for all students (optimized - only check recent assignments)
            students = self.get_students()
            assignments = self.get_assignments()[:limit_assignments]  # Limit to avoid too many calls

            print(f"Checking activity for {len(students)} students across {len(assignments)} assignments...")

            for student in students:
                student_id = student["id"]
                activity_data[student["name"]] = {
                    "total_assignments_checked": len(assignments),
                    "submissions": 0,
                    "last_activity": None,
                }

                for assignment in assignments:
                    try:
                        submissions = self.get_submissions(assignment["id"])
                        user_sub = [s for s in submissions if s.get("user_id") == student_id]
                        if user_sub and user_sub[0].get("submitted_at"):
                            activity_data[student["name"]]["submissions"] += 1
                            submitted_at = user_sub[0].get("submitted_at")
                            if (
                                not activity_data[student["name"]]["last_activity"]
                                or submitted_at > activity_data[student["name"]]["last_activity"]
                            ):
                                activity_data[student["name"]]["last_activity"] = submitted_at
                    except Exception:
                        continue

        return activity_data

    def get_roll_call_attendance(self) -> Optional[Dict[str, Any]]:
        """
        Attempt to get Roll Call Attendance data.

        Roll Call Attendance is an LTI tool. This function tries various
        endpoints that might work depending on how it's configured.

        Returns:
        --------
        dict or None
            Attendance data if available
        """
        # First, find the Roll Call Attendance tool
        try:
            r = requests.get(
                f"https://canvas.pitt.edu/api/v1/courses/{self.course_id}/external_tools",
                headers=self.header,
            )
            r.raise_for_status()
            tools = r.json()

            roll_call_tool = None
            for tool in tools:
                if "roll" in tool.get("name", "").lower() and "call" in tool.get("name", "").lower():
                    roll_call_tool = tool
                    break

            if not roll_call_tool:
                return None

            tool_id = roll_call_tool.get("id")
            print(f"Found Roll Call Attendance tool (ID: {tool_id})")

            # Try various endpoints that might work
            endpoints_to_try = [
                f"external_tools/{tool_id}/attendance",
                f"external_tools/{tool_id}/roll_call",
                f"external_tools/{tool_id}/sessions",
                f"lti/courses/{self.course_id}/attendance",
                f"lti/courses/{self.course_id}/roll_call",
            ]

            for endpoint in endpoints_to_try:
                try:
                    r = requests.get(
                        f"https://canvas.pitt.edu/api/v1/courses/{self.course_id}/{endpoint}",
                        headers=self.header,
                    )
                    if r.status_code == 200:
                        return r.json()
                except Exception:
                    continue

            # If direct API doesn't work, return tool info so user knows it exists
            return {
                "tool_found": True,
                "tool_name": roll_call_tool.get("name"),
                "tool_id": tool_id,
                "note": "Roll Call Attendance tool found but API endpoint not accessible. "
                "You may need to export attendance data manually from Canvas or "
                "check with your Canvas admin for custom API access.",
            }

        except Exception as e:
            print(f"Error accessing Roll Call Attendance: {e}")
            return None

    def get_enrollments(self) -> Any:
        """Get all enrollments in the course (includes enrollment dates, status, etc.)."""
        return self.get("enrollments")


def canvas_client_from_env() -> CanvasClient:
    token = os.getenv("CANVAS_TOKEN")
    course_id = os.getenv("CANVAS_COURSE_ID")
    if not token or not course_id:
        raise ValueError("Missing required env vars: CANVAS_TOKEN and CANVAS_COURSE_ID")
    return CanvasClient(token=token, course_id=course_id)
