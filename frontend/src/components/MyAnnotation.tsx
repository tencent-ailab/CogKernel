import { FloatButton } from "antd";
import { useState, useContext, useEffect } from "react";
import { Progress, Modal, Form, DatePicker, message, Button } from "antd";
import {
  EditOutlined,
  DownloadOutlined,
  CloseCircleOutlined,
  DatabaseOutlined,
  ToolOutlined,
  ExperimentOutlined,
} from "@ant-design/icons";
import axios from "axios";
import dayjs, { Dayjs } from "dayjs";
import { StatusContext } from "../contexts/StatusContext";
import {
  LabelList,
  BarChart,
  Bar,
  Rectangle,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import html2canvas from "html2canvas";

interface AnnotationType {
  session_id: string;
  message_id: string;
  username: string;
  tag: string;
  for_evaluation: boolean;
  old_message: string;
  suggestion: string;
  annotations: string;
  created_time: string;
  updated_time: string;
}

function toLocalISOString(date: Date): string {
  const offset = date.getTimezoneOffset() * 60000;
  const localDate = new Date(date.getTime() - offset);
  const iso = localDate.toISOString();
  return iso;
}
const AnnotationButton = () => {
  const [visible, setVisible] = useState(false);
  const [form] = Form.useForm();
  const context = useContext(StatusContext);

  if (!context) {
    return <div>Context not available.</div>;
  }

  const { CKStatus, setCKStatus } = context;

  const showModal = () => {
    form.setFieldsValue({
      startDate: dayjs(),
      endDate: dayjs(),
    });
    setVisible(true);
  };

  const handleCancel = () => {
    setVisible(false);
  };

  const handleDownload = (downloadType: string) => {
    form
      .validateFields()
      .then((values) => {
        console.log("Received values of form: ", values);
        const username = localStorage.getItem("username");
        const startDate = values.startDate.format("YYYY-MM-DD");
        const endDate = values.endDate.format("YYYY-MM-DD");
        console.log("downloadType", downloadType);

        axios
          .get("/api/download_annotations", {
            params: {
              username: username,
              start_date: startDate,
              end_date: endDate,
              download_type: downloadType,
            },
          })
          .then((response) => {
            const fileName = `${username}_${startDate}_${endDate}_${downloadType}.jsonl`;
            console.log(response.data);
            let jsonLines = "";
            if (downloadType === "annotation") {
              jsonLines = response.data
                .map((obj: AnnotationType) => JSON.parse(obj["annotations"]))
                .map((obj: JSON) => JSON.stringify(obj, null, 0))
                .join("\n");
            } else if (downloadType === "all") {
              jsonLines = response.data
                .map((obj: AnnotationType) =>
                  JSON.stringify(
                    {
                      session_id: obj["session_id"],
                      message_id: obj["message_id"],
                      username: obj["username"],
                      tag: obj["tag"],
                      for_evaluation: obj["for_evaluation"],
                      old_data: JSON.parse(obj["old_message"]),
                      suggestion: obj.suggestion,
                      annotations: JSON.parse(obj["annotations"]),
                      created_time: obj["created_time"],
                      updated_time: obj["updated_time"],
                    },
                    null,
                    0
                  )
                )
                .join("\n");
            }
            const blob = new Blob([jsonLines], { type: "application/jsonl" });
            const href = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = href;
            link.download = fileName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            message.success("Download successfully!");
          })
          .catch((err) => {
            console.error("Download Failed", err);
            message.error("Download Failed");
          })
          .finally(() => {
            setVisible(false);
          });
      })
      .catch((info) => {
        console.log("Validate Failed:", info);
      });
  };
  const disabledStartDate = (current: Dayjs) => {
    return current && current > dayjs().endOf("day");
  };
  const disabledEndDate = (current: Dayjs) => {
    return current && current > dayjs().endOf("day");
  };

  const activateOnlineAnnotation = () => {
    console.log("activateOnlineAnnotation");
    setCKStatus((prevStatus) => ({
      ...prevStatus,
      activated_functions_online_annotation: true,
    }));
  };

  const deactivateOnlineAnnotation = () => {
    console.log("deactivateOnlineAnnotation");
    setCKStatus((prevStatus) => ({
      ...prevStatus,
      activated_functions_online_annotation: false,
    }));
  };

  const checkAnnotationStatistics = () => {
    console.log("checkAnnotationStatistics");
    axios
      .get("/api/annotation_statistics", {
        params: {
          username: localStorage.getItem("username"),
          current_time: toLocalISOString(new Date()),
        },
      })
      .then((response) => {
        // console.log(response.data);
        message.info(
          `Annotation Statistics (Today): ${response.data.data["today"]}
           Annotation Statistics (This week): ${response.data.data["this_week"]}
           Annotation Statistics (All): ${response.data.data["total"]}`,
          5
        );
      })
      .catch((err) => {
        console.error("Failed to get the annotation statistics", err);
        message.error("Failed to get the annotation statistics");
      });
  };

  return (
    <>
      <FloatButton.Group
        trigger="hover"
        shape="circle"
        type="default"
        style={{ position: "fixed", right: 40, bottom: 40 }}
        icon={<ToolOutlined />}
      >
        <FloatButton
          type="default"
          shape="circle"
          tooltip="Check Annotation Statistics"
          icon={<DatabaseOutlined />}
          onClick={checkAnnotationStatistics}
        />
        <FloatButton
          type="default"
          shape="circle"
          tooltip="Download Data"
          icon={<DownloadOutlined />}
          onClick={showModal}
        />
        {CKStatus.activated_functions_online_annotation ? (
          <FloatButton
            type="primary"
            shape="circle"
            tooltip="Disable Online Annotation"
            icon={<CloseCircleOutlined />}
            onClick={deactivateOnlineAnnotation}
          />
        ) : (
          <FloatButton
            type="default"
            shape="circle"
            tooltip="Enable Online Annotation"
            icon={<EditOutlined />}
            onClick={activateOnlineAnnotation}
          />
        )}
      </FloatButton.Group>
      <Modal
        title="Download Annotation Data"
        open={visible}
        closable={true}
        onCancel={handleCancel}
        footer={[
          <Button key="back" onClick={handleCancel}>
            Cancel
          </Button>,
          <Button
            key="downloadCore"
            type="primary"
            onClick={() => handleDownload("annotation")}
          >
            Get My Annotation
          </Button>,
          <Button
            key="downloadRaw"
            type="primary"
            onClick={() => handleDownload("all")}
          >
            Get All Annotation
          </Button>,
        ]}
      >
        <Form form={form} layout="vertical" name="form_in_modal">
          <Form.Item
            name="startDate"
            label="Start time"
            rules={[
              { required: true, message: "Please choose the start time" },
            ]}
          >
            <DatePicker
              defaultValue={dayjs()}
              disabledDate={disabledStartDate}
            />
          </Form.Item>
          <Form.Item
            name="endDate"
            label="End time"
            dependencies={["startDate"]}
            rules={[
              { required: true, message: "Please choose the end time" },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (
                    !value ||
                    getFieldValue("startDate").isBefore(value) ||
                    getFieldValue("startDate").isSame(value, "day")
                  ) {
                    return Promise.resolve();
                  }
                  return Promise.reject(
                    new Error("The end time must be after the start time.")
                  );
                },
              }),
            ]}
          >
            <DatePicker defaultValue={dayjs()} disabledDate={disabledEndDate} />
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
};

const DeveloperAccess = () => {
  const context = useContext(StatusContext);
  if (!context) {
    return <div>Context not available.</div>;
  }
  const { CKStatus, setCKStatus } = context;
  const [isAllowed, setIsAllowed] = useState(false);
  const checkUser = async () => {
    const username = CKStatus.username;
    if (!username) {
      setIsAllowed(false);
      return;
    }
    if (username) {
      try {
        const response = await axios.get("/api/auth/check_developer", {
          params: { username: username },
        });
        if (response.status === 200) {
          if (response.data.is_developer) {
            setIsAllowed(true);
          } else {
            setIsAllowed(false);
          }
        } else if (response.status === 401) {
          setIsAllowed(false);
        }
      } catch (error) {
        console.error("Error checking user permission:", error);
        setIsAllowed(false);
      }
    }
  };
  useEffect(() => {
    checkUser();
  }, [CKStatus.username]);
  if (isAllowed) {
    return (
      <StatusContext.Provider value={{ CKStatus, setCKStatus }}>
        <AnnotationButton />
      </StatusContext.Provider>
    );
  } else {
    return null;
  }
};

export default DeveloperAccess;
